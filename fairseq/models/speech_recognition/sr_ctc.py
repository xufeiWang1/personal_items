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

from fairseq.models.speech_recognition import Conv1dSubsampler
from fairseq.models.speech_recognition.transformer_encoder import SRTransformerEncoder
from fairseq.models.speech_recognition.conformer_encoder import SRConformerEncoder

@dataclass
class SRCTCConfig(FairseqDataclass):

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

    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
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
