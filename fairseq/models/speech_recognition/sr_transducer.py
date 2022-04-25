#!/usr/bin/env python3

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

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


class Conv1dSubsampler(nn.Module):
    """Convolutional subsampler: a stack of 1D convolution (along temporal
    dimension) followed by non-linear activation via gated linear units
    (https://arxiv.org/abs/1911.08460)

    Args:
        in_channels (int): the number of input channels
        mid_channels (int): the number of intermediate channels
        out_channels (int): the number of output channels
        kernel_sizes (List[int]): the kernel size for each convolutional layer
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_sizes: List[int] = (3, 3),
    ):
        super(Conv1dSubsampler, self).__init__()
        self.n_layers = len(kernel_sizes)
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )

    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, src_tokens, src_lengths):
        bsz, in_seq_len, _ = src_tokens.size()  # B x T x (C x D)
        x = src_tokens.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        _, _, out_seq_len = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        return x, self.get_out_seq_lens_tensor(src_lengths)


@dataclass
class SRTransducerConfig(FairseqDataclass):

    #### downsample layer config
    conv_kernel_sizes: str = "5,5"
    conv_channels: int = 1024
    input_feat_per_channel: int = 80
    input_channels: int = 1

    #### encoder config
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
        encoder = SRTransformerEncoder(args)
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
        jointnet_out = self.jointnet(encoder_out, decoder_out)
        return jointnet_out, encoder_out_lengths


class SRTransformerEncoder(FairseqEncoder):
    """Speech-to-text Transformer encoder that consists of input subsampler and
    Transformer encoder."""

    def __init__(self, args):
        super().__init__(None)

        self.encoder_freezing_updates = args.encoder_freezing_updates
        self.num_updates = 0

        self.dropout_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0
        self.padding_idx = 1

        self.subsample = Conv1dSubsampler(
            args.input_feat_per_channel * args.input_channels,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")],
        )

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, args.encoder_embed_dim, self.padding_idx
        )

        self.transformer_layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        x, input_lengths = self.subsample(src_tokens, src_lengths)
        x = self.embed_scale * x

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)
        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = self.dropout_module(x)

        encoder_states = []

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_out_lengths": [input_lengths],
            "encoder_padding_mask": [encoder_padding_mask]
            if encoder_padding_mask.any()
            else [],  # B x T
            "encoder_embedding": [],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    def forward(self, src_tokens, src_lengths, return_all_hiddens=False):
        if self.num_updates < self.encoder_freezing_updates:
            with torch.no_grad():
                x = self._forward(
                    src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
                )
        else:
            x = self._forward(
                src_tokens, src_lengths, return_all_hiddens=return_all_hiddens
            )
        return x

    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

# TODO
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

    # encoder_out: T x B x C
    # decoder_out: L x B X C
    def forward(self, encoder_out, decoder_out, apply_output_layer=True):
        encoder_out = self.laynorm_proj_encoder(self.proj_encoder(encoder_out)).transpose(0, 1)
        decoder_out = self.laynorm_proj_decoder(self.proj_decoder(decoder_out)).transpose(0, 1)
        out = nn.functional.relu(encoder_out.unsqueeze(2) + decoder_out.unsqueeze(1))
        if apply_output_layer:
            out = self.fc_out(out)
        return out

