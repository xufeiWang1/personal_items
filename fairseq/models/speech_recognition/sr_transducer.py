#!/usr/bin/env python3

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from fairseq.incremental_decoding_utils import with_incremental_state
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import LSTMEncoder
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.transformer import Embedding, TransformerEncoder
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer,
)

logger = logging.getLogger(__name__)

from fairseq.models.speech_recognition import Conv1dSubsampler
from fairseq.models.speech_recognition.transformer_encoder import SRTransformerEncoder
from fairseq.models.speech_recognition.conformer_encoder import SRConformerEncoder


@dataclass
class SRTransducerConfig(FairseqDataclass):

    #### downsample layer config
    conv_kernel_sizes: str = "5,5"
    conv_channels: int = 1024
    input_feat_per_channel: int = 80
    input_channels: int = 1

    #### encoder config
    encoder_type: str = "transformer"
    encoder_embed_dim: int = 256
    encoder_ffn_embed_dim: int = 256 * 8
    encoder_attention_heads: int = 4
    dropout: float = 0.1
    encoder_freezing_updates: int = 0
    encoder_layers: int = 12
    encoder_normalize_before: bool = True
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: str = "relu"
    no_token_positional_embeddings: bool = False
    layernorm_embedding: bool = False
    load_pretrained_encoder_from: Optional[str] = None
    adaptive_input: bool = False
    no_scale_embedding: bool = False
    max_source_positions: int = 6000
    # for conformer encoder
    pos_enc_type: str = "rel_pos"
    depthwise_conv_kernel_size: int = 31
    attn_type: Optional[str] = "espnet"
    fp16: bool = False

    #### decoder config
    # decoder_type: str = "transformer"
    decoder_type: str = "lstm"

    # transformer
    decoder_attention_heads: int = 4
    decoder_embed_dim: int = 256
    decoder_ffn_embed_dim: int = 256*8
    decoder_layers: int = 6
    decoder_normalize_before: bool = True
    decoder_learned_pos: bool = False
    share_decoder_input_output_embed: bool = False
    decoder_layerdrop: float = 0.0
    decoder_output_dim: int = 256

    # lstm
    decoder_hidden_size: int = 256
    decoder_num_layers: int  = 2
    decoder_dropout_in: float = 0.1
    decoder_dropout_out: float = 0.1

    #### jointnet config
    joint_dim: int = 256
    adaptive_softmax_cutoff: Optional[int] = None
    adaptive_softmax_dropout: float = 0

    #### other config
    speaker_to_id: Optional[int] = 1
    quant_noise_pq: int = 0



@register_model("sr_transducer", dataclass=SRTransducerConfig)
class S2TTransducerModel(BaseFairseqModel):
    """Adapted Transformer model (https://arxiv.org/abs/1706.03762) for
    speech-to-text tasks. The Transformer encoder/decoder remains the same.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, decoder, jointnet):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.jointnet = jointnet

    @classmethod
    def build_encoder(cls, args):
        if args.encoder_type == "transformer":
            encoder = SRTransformerEncoder(args)
        elif args.encoder_type == "conformer":
            encoder = SRConformerEncoder(args)
        else:
            raise NotImplemented
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None:
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        if args.decoder_type.lower() == "transformer":
            return TransducerTransformerDecoder(args, task.target_dictionary, embed_tokens)
        elif args.decoder_type.lower() == "lstm":
            return TransducerLSTMDecoder(args, task.target_dictionary, embed_tokens)
        else:
            raise RuntimeError(f"Unknow Transducer Decoder type: {args.decoder_type}")


    @classmethod
    def build_jointnet(cls, args, task):
        return JointNet(args, task)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        jointnet = cls.build_jointnet(args, task)
        return cls(encoder, decoder, jointnet)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        encoder_outs = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        encoder_out = encoder_outs['encoder_out'][0]
        encoder_out_lengths = encoder_outs['encoder_out_lengths'][0]
        # decoder_outs = self.decoder(src_tokens=prev_output_tokens, src_lengths=target_lengths)

        # current rnnt_loss requires the out dim: B x T x (L+1) x D
        # padding <eos> at the end of each batch
        pad_output_tokens = prev_output_tokens.new(prev_output_tokens.size(0), 1).fill_(2)
        full_output_tokens = torch.cat((prev_output_tokens, pad_output_tokens),dim=1)
        # decoder_outs = self.decoder.forward_scriptable(src_tokens=full_output_tokens, src_lengths=src_lengths)
        decoder_outs = self.decoder.forward_scriptable(src_tokens=full_output_tokens)
        # decoder_outs = self.decoder(src_tokens=prev_output_tokens)
        decoder_out = decoder_outs['encoder_out'][0]

        encoder_out = encoder_out.transpose(0, 1)
        decoder_out = decoder_out.transpose(0, 1)

        jointnet_out = self.jointnet(encoder_out, decoder_out)
        return jointnet_out, encoder_out_lengths


@with_incremental_state
class TransducerLSTMDecoder(LSTMEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary, args.decoder_embed_dim,
                         args.decoder_hidden_size, args.decoder_num_layers,
                         args.decoder_dropout_in, args.decoder_dropout_out,
                         )


    def forward_scriptable(self,
                src_tokens,
                src_lengths: Optional[torch.Tensor] = None,
                enforce_sorted: bool=False,):
        x, final_hiddens, final_cells, encoder_padding_mask = self.forward(src_tokens, src_lengths, enforce_sorted)

        return {
            "encoder_out": [x],  # T x B x C
            "final_hiddens": [final_hiddens], # num_layer x B x C
            "final_cells": [final_cells], # num_layer x B x C
            "encoder_padding_mask": [encoder_padding_mask] # seq_len x batch
        }

    def extract_features(self,
                         prev_output_tokens,
                         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
                         **unused,):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - attention weights of shape `(batch, tgt_len, src_len)`
        """

        if (
            incremental_state is not None
            and self._get_full_incremental_state_key("cached_state")
            in incremental_state
        ):
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        # x = self.embed_tokens(prev_output_tokens)
        # x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        # x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        if (
            incremental_state is not None
            and self._get_full_incremental_state_key("cached_state")
            in incremental_state
        ):
            prev_hiddens, prev_cells = self.get_cached_state(
                incremental_state
            )
        else:
            # zero_state = prev_output_tokens.new_zeros(bsz, self.hidden_size)
            # prev_hiddens = [zero_state for i in range(self.num_layers)]
            # prev_cells = [zero_state for i in range(self.num_layers)]
            prev_hiddens = prev_output_tokens.new_zeros((self.num_layers, bsz, self.hidden_size), dtype=torch.float)
            prev_cells = prev_output_tokens.new_zeros((self.num_layers, bsz, self.hidden_size), dtype=torch.float)

        # outs = []

        x, hiddens, cells, _ = self.forward(prev_output_tokens, h0=prev_hiddens, c0=prev_cells)

        '''
        for j in range(seqlen):
            input = x[j, :, :]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                input = hidden
                input = self.dropout_out_module(input)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # save final output
            outs.append(input)
        '''

        # Stack all the necessary tensors together and store
        # prev_hiddens_tensor = torch.stack(prev_hiddens)
        # prev_cells_tensor = torch.stack(prev_cells)
        """
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
            },
        )
        """
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": hiddens,
                "prev_cells": cells,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)

        # collect outputs across time steps
        # x = torch.cat(outs, dim=0).view(seqlen, bsz, -1)
        # assert x.size(2) == self.hidden_size
        # x = outs

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        attn_scores = None

        return x, attn_scores

    def get_cached_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
    ) -> Tuple[List[Tensor], List[Tensor], Optional[Tensor]]:
        cached_state = self.get_incremental_state(incremental_state, "cached_state")
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        # assert prev_cells_ is not None
        # prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        # prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        return prev_hiddens_, prev_cells_


    # used for greedy search
    def initialize_cached_state(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        bsz = prev_output_tokens.size(0)
        x = self.embed_tokens(prev_output_tokens)
        zero_states = x.new_zeros(self.num_layers, bsz, self.hidden_size)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": zero_states,
                "prev_cells": zero_states,
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cache_state)


    def masked_copy_cached_state(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        src_cached_state: Tuple[Optional[Union[List[torch.Tensor], torch.Tensor]]],
        mask: Tensor,
    ):
        if (
            incremental_state is None
            or self._get_full_incremental_state_key("cached_state")
            not in incremental_state
        ):
            assert src_cached_state is None or len(src_cached_state) == 0
            return
        prev_hiddens, prev_cells = self.get_cached_state(incremental_state)
        src_prev_hiddens, src_prev_cells = (
            src_cached_state[0],
            src_cached_state[1],
        )

        def masked_copy_state(state: Optional[Tensor], src_state: Optional[Tensor]):
            if state is None:
                assert src_state is None
                return None
            else:
                assert (
                    state.size(0) == mask.size(0)
                    and src_state is not None
                    and state.size() == src_state.size()
                )
                state[mask, ...] = src_state[mask, ...]
                return state

        prev_hiddens = [
            masked_copy_state(p, src_p)
            for (p, src_p) in zip(prev_hiddens, src_prev_hiddens)
        ]
        prev_cells = [
            masked_copy_state(p, src_p)
            for (p, src_p) in zip(prev_cells, src_prev_cells)
        ]

        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
            },
        )
        self.set_incremental_state(incremental_state, "cached_state", cached_state_new)




class TransducerTransformerDecoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens, return_fc=False)


    # need to apply attn_mask
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []
        fc_results = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        src_length = x.size(0)
        attn_mask = torch.triu(utils.fill_with_neg_inf(torch.zeros([src_length, src_length])), diagonal=1).to(x.device)
        for layer in self.layers:
            lr = layer(
                x, encoder_padding_mask=encoder_padding_mask if has_pads else None, attn_mask=attn_mask
            )

            if isinstance(lr, tuple) and len(lr) == 2:
                x, fc_result = lr
            else:
                x = lr
                fc_result = None

            if return_all_hiddens and not torch.jit.is_scripting():
                assert encoder_states is not None
                encoder_states.append(x)
                fc_results.append(fc_result)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        src_lengths = (
            src_tokens.ne(self.padding_idx)
            .sum(dim=1, dtype=torch.int32)
            .reshape(-1, 1)
            .contiguous()
        )
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "fc_results": fc_results,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
        }


class JointNet(nn.Module):
    def __init__(self, cfg, task):
        super().__init__()
        self.joint_dim = cfg.joint_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        self.decoder_embed_dim = cfg.decoder_embed_dim
        # add blank symbol in output layer
        self.out_dim = len(task.target_dictionary)
        self.proj_encoder = nn.Linear(self.encoder_embed_dim, self.joint_dim)
        self.laynorm_proj_encoder = LayerNorm(self.joint_dim)
        self.proj_decoder = nn.Linear(self.decoder_embed_dim, self.joint_dim)
        self.laynorm_proj_decoder = LayerNorm(cfg.joint_dim)

        self.fc_out = nn.Linear(self.joint_dim, self.out_dim)
        nn.init.normal_(self.fc_out.weight, mean=0, std=self.joint_dim**-0.5)

        self.onnx_trace = False
        self.adaptive_softmax = None

    # encoder_out: B x T x C
    # decoder_out: B X U x C
    def forward(self, encoder_out, decoder_out, apply_output_layer=True):
        encoder_out = self.laynorm_proj_encoder(self.proj_encoder(encoder_out))
        decoder_out = self.laynorm_proj_decoder(self.proj_decoder(decoder_out))
        out = nn.functional.relu(encoder_out.unsqueeze(2) + decoder_out.unsqueeze(1))
        if apply_output_layer:
            out = self.fc_out(out)
        return out


    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool = True,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
