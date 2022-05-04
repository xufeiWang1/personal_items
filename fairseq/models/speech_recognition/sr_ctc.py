#!/usr/bin/env python3

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
class SRCTCConfig(FairseqDataclass):

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

    outproj_dim: int = 512
    adaptive_softmax_cutoff: Optional[int] = None
    adaptive_softmax_dropout: float = 0

    #### other config
    speaker_to_id: Optional[int] = 1
    quant_noise_pq: int = 0



@register_model("sr_ctc", dataclass=SRCTCConfig)
class S2TCTCModel(BaseFairseqModel):
    """CTC model (https://www.cs.toronto.edu/~graves/icml_2006.pdf) for
    speech-to-text tasks. The CTC encoder consists of TransformerEncoder.
    A trainable input subsampler is prepended to the Transformer encoder to
    project inputs into the encoder dimension as well as downsample input
    sequence for computational efficiency."""

    def __init__(self, encoder, outlayer):
        super().__init__()
        self.encoder = encoder
        self.outlayer = outlayer

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
    def build_outlayer(cls, args, task):
        return OutputLayer(args, task)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        # base_architecture(args)

        encoder = cls.build_encoder(args)
        outlayer= cls.build_outlayer(args, task)
        return cls(encoder, outlayer)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        # lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs = self.outlayer.get_normalized_probs_scriptable(net_output, log_probs, sample)
        lprobs.batch_first = True
        return lprobs

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        # acoustic encoder
        encoder_outs = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths)
        encoder_out = encoder_outs['encoder_out'][0]
        encoder_out_lengths = encoder_outs['encoder_out_lengths'][0]

        # output classify layer
        outs = self.outlayer(encoder_out)

        return outs, encoder_out_lengths


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



class OutputLayer(nn.Module):
    def __init__(self, cfg, task):
        super().__init__()
        self.outproj_dim = cfg.outproj_dim
        self.encoder_embed_dim = cfg.encoder_embed_dim
        # add blank symbol in output layer
        self.out_dim = len(task.target_dictionary)
        self.proj_encoder = nn.Linear(self.encoder_embed_dim, self.outproj_dim)
        self.laynorm_proj_encoder = LayerNorm(self.outproj_dim)

        self.fc_out = nn.Linear(self.outproj_dim, self.out_dim)
        nn.init.normal_(self.fc_out.weight, mean=0, std=self.outproj_dim**-0.5)

        self.onnx_trace = False
        self.adaptive_softmax = None

    # encoder_out: B x T x C
    def forward(self, encoder_out, apply_output_layer=True):
        encoder_out = self.laynorm_proj_encoder(self.proj_encoder(encoder_out))
        if apply_output_layer:
            out = self.fc_out(encoder_out)
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
