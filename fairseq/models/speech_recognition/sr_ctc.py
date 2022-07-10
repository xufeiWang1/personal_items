#!/usr/bin/env python3

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

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
    # conv_kernel_sizes: str = "5"
    conv_channels: int = 1024
    input_feat_per_channel: int = 80
    input_channels: int = 1

    #### encoder output pooling layer config
    use_encoder_output_subsampler: bool = False
    pool_kernel_size: int = 2
    pool_stride_size: int = 2
    pool_padding_size: int = 0
    # conv2d or superframe
    subsample_type: str = "conv2d"

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
    # for pretrained SSL model
    ssl_model_path: Optional[str] = None
    randinit: bool = False


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

    def __init__(self, encoder, outlayer, args=None):
        super().__init__()
        self.encoder = encoder
        self.outlayer = outlayer
        self.encoder_type = args.encoder_type

        if args.encoder_type in ["data2vec_v2", "hubert_v2", "hubert", "hubert_feadaptor", "data2vec_feadaptor", "wav2vec2_feadaptor"] :
            from fairseq.models.speech_recognition.convsubsampler import Conv1dSubsampler, Pooling1DSubsampler, SuperFrame

            if args.encoder_type in ["hubert_feadaptor", "data2vec_feadaptor", "wav2vec2_feadaptor"]:
                args.input_feat_per_channel = 80
                args.input_channels = 1
            self.subsample = Conv1dSubsampler(
                args.input_feat_per_channel * args.input_channels,
                args.conv_channels,
                args.encoder_embed_dim,
                [int(k) for k in args.conv_kernel_sizes.split(",")] )

            self.linear = torch.nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
            nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
            self.dropout = torch.nn.Dropout(args.dropout)

            # global cmvn
            # stats_npz_path = "/mnt/data/librispeech/data4fairseq-s-rawwav/global_cmvn.npy"
            stats_npz_path = "/mnt/data/GigaSpeech/tempdir/global_cmvn.npy"
            stats = np.load(stats_npz_path, allow_pickle=True).tolist()
            self.mean, self.std = stats["mean"], stats["std"]

            # specaug
            specaug_config = {"freq_mask_F": 30, "freq_mask_N": 2, "time_mask_N": 2, "time_mask_T": 40, "time_mask_p": 1.0, "time_wrap_W": 0}
            from fairseq.data.audio.feature_transforms.specaugment import SpecAugmentTransform
            self.specaug_transform = SpecAugmentTransform.from_config_dict(specaug_config)

        self._step = 0

    @classmethod
    def build_encoder(cls, args):
        if args.encoder_type == "transformer":
            encoder = SRTransformerEncoder(args)
        elif args.encoder_type == "conformer":
            encoder = SRConformerEncoder(args)
        # load pretrained SSL model
        elif args.encoder_type in ["data2vec", "data2vec_v2", "data2vec_feadaptor"]:

            from examples.data2vec.models.data2vec_audio import Data2VecAudioModel, Data2VecAudioConfig
            from fairseq import checkpoint_utils
            ssl_model_path = args.ssl_model_path
            state = checkpoint_utils.load_checkpoint_to_cpu(ssl_model_path)
            model_file_cfg = state["cfg"]["model"]
            ssl_model_cfg = Data2VecAudioConfig()
            # update the config class for data2vec model from the checkpoint file
            for k, v in model_file_cfg.items():
                setattr(ssl_model_cfg, k, v)
            encoder = Data2VecAudioModel(ssl_model_cfg)
            if "_ema" in state["model"]:
                del state["model"]["_ema"]
            # fix the name mismatch in checkpoint from Fiarseq github repo
            if state["model"].get("final_proj.0.weight", None) is not None:
                state["model"]["final_proj.weight"] = state["model"].pop("final_proj.0.weight")
                state["model"]["final_proj.bias"] = state["model"].pop("final_proj.0.bias")
            if args.randinit is False:
                encoder.load_state_dict(state["model"], strict=True)

        elif args.encoder_type in ["hubert", "hubert_v2", "hubert_feadaptor"]:
            from fairseq.models.hubert import HubertConfig, HubertModel
            from fairseq import checkpoint_utils
            ssl_model_path = args.ssl_model_path
            state = checkpoint_utils.load_checkpoint_to_cpu(ssl_model_path)
            model_file_cfg = state["cfg"]["model"]
            ssl_model_cfg = HubertConfig()
            # update the config class for data2vec model from the checkpoint file
            for k, v in model_file_cfg.items():
                setattr(ssl_model_cfg, k, v)
            encoder = HubertModel(ssl_model_cfg)
            state["model"].pop("label_embs_concat", None)
            if args.randinit is False:
                encoder.load_state_dict(state["model"], strict=True)
        elif args.encoder_type in ["wav2vec2", "wav2vec2_feadaptor"]:
            from fairseq.models.wav2vec import Wav2Vec2Model, Wav2Vec2Config
            from fairseq import checkpoint_utils
            ssl_model_path = args.ssl_model_path
            state = checkpoint_utils.load_checkpoint_to_cpu(ssl_model_path)
            model_file_cfg = state["cfg"]["model"]
            ssl_model_cfg = Wav2Vec2Config()
            for k, v in model_file_cfg.items():
                setattr(ssl_model_cfg, k, v)
            encoder = Wav2Vec2Model(ssl_model_cfg)
            if args.randinit is False:
                encoder.load_state_dict(state["model"], strict=True)
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
        return cls(encoder, outlayer, args)

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


    def extract_fbank_features(self, source, src_lengths):
        from fairseq.data.audio.audio_utils import _get_torchaudio_fbank, _get_kaldi_fbank
        sample_rate = 16000
        n_mel_bins = 80

        fbank_lengths = []
        fbank_features = []
        data_dtype = source.dtype
        with torch.no_grad():
            source = source.float()
            for batch_idx in range(source.size(0)):
                _waveform = source[batch_idx][:src_lengths[batch_idx]]
                _waveform = _waveform * (2 ** 15)
                _waveform = _waveform.float().cpu().unsqueeze(0).numpy()
                features = _get_kaldi_fbank(_waveform, sample_rate, n_mel_bins)
                if features is None:
                    features = _get_torchaudio_fbank(_waveform, sample_rate, n_mel_bins)

                features = torch.from_numpy(features)
                features = np.subtract(features, self.mean)
                features = np.divide(features, self.std)
                features = features.cuda()

                feat_len = features.size(0)
                if batch_idx == 0:
                    max_len  = feat_len
                else:
                    if feat_len != max_len:
                        pad_len = max_len - feat_len
                        features_padding = features.new(pad_len, n_mel_bins).fill_(0)
                        features = torch.cat([features, features_padding], dim=0)
                        features = features.type(source.dtype)
                # only apply specaug during Training
                if self.encoder.training is True:
                    features = self.specaug_transform(features)

                fbank_features.append(features)
                fbank_lengths.append(feat_len)

            fbank_features = torch.stack(fbank_features, dim=0).contiguous().type(data_dtype)
            fbank_lengths = torch.Tensor(fbank_lengths).int().cuda()

        # subsampling on fbank features
        fbank_features, encoder_out_lengths = self.subsample(fbank_features, src_lengths=fbank_lengths)
        fbank_features = self.linear(fbank_features)
        fbank_features = self.dropout(fbank_features)
        return fbank_features, encoder_out_lengths


    def ComputeFrontEndMSELoss(self, fbank_features, fbank_lengths, rawwav_features, rawwav_lengths):
        l2loss = 0
        mbsize = fbank_lengths.size(0)

        for index in range(mbsize):
            if self.subsample.n_layers == 2:
                valid_len = min(fbank_lengths[index], rawwav_lengths[index]//2)
                l2loss += torch.sum(torch.pow(fbank_features[index,:valid_len, :]-rawwav_features[index,:valid_len*2:2,:], 2)) / (valid_len * fbank_features.size(-1))
            else:
                valid_len = min(fbank_lengths[index], rawwav_lengths[index])
                l2loss += torch.sum(torch.pow(fbank_features[index,:valid_len, :]-rawwav_features[index,:valid_len,:], 2)) / (valid_len * fbank_features.size(-1))

        l2loss /= mbsize
        # l2loss = torch.sum(torch.pow(fbank_features-rawwav_features, 2)) / torch.numel(fbank_features)


        return l2loss


    def forward(self, src_tokens, src_lengths, prev_output_tokens=None):
        """
        The forward method inherited from the base class has a **kwargs
        argument in its input, which is not supported in torchscript. This
        method overwrites the forward method definition without **kwargs.
        """
        # acoustic encoder
        if self.encoder_type == "data2vec":
            # B x T x 1
            source = src_tokens.squeeze(-1)
            # postprocessing, normalize for data2vec
            with torch.no_grad():
                source = nn.functional.layer_norm(source, source.shape[1:])

            encoder_padding_mask = lengths_to_padding_mask(src_lengths)
            res = self.encoder.extract_features(source, encoder_padding_mask)
            encoder_out = res["x"]
            padding_mask = res["padding_mask"]
            if padding_mask is None:
                B = encoder_out.size(0)
                max_T = encoder_out.size(1)
                encoder_out_lengths = torch.zeros(B, dtype=torch.int32, device=source.device).fill_(max_T)
            else:
                encoder_out_lengths = torch.sum(~padding_mask, dim=-1)
            # encoder out dim: B x T x C -> T x B x C
            encoder_out = encoder_out.transpose(0, 1)

        elif self.encoder_type == "hubert":
            # B x T x 1
            source = src_tokens.squeeze(-1)
            # postprocessing, normalize for data2vec
            with torch.no_grad():
                source = nn.functional.layer_norm(source, source.shape[1:])

            encoder_padding_mask = lengths_to_padding_mask(src_lengths)

            encoder_out, padding_mask = self.encoder.extract_features(source, encoder_padding_mask)
            if padding_mask is None:
                B = encoder_out.size(0)
                max_T = encoder_out.size(1)
                encoder_out_lengths = torch.zeros(B, dtype=torch.int32, device=source.device).fill_(max_T)
            else:
                encoder_out_lengths = torch.sum(~padding_mask, dim=-1)
            # encoder out dim: B x T x C -> T x B x C
            encoder_out = encoder_out.transpose(0, 1)

        elif self.encoder_type == "wav2vec2":
            # B x T x 1
            source = src_tokens.squeeze(-1)
            # postprocessing, normalize for data2vec
            with torch.no_grad():
                source = nn.functional.layer_norm(source, source.shape[1:])

            encoder_padding_mask = lengths_to_padding_mask(src_lengths)

            res = self.encoder.extract_features(source, encoder_padding_mask)
            encoder_out = res["x"]
            padding_mask = res["padding_mask"]
            if padding_mask is None:
                B = encoder_out.size(0)
                max_T = encoder_out.size(1)
                encoder_out_lengths = torch.zeros(B, dtype=torch.int32, device=source.device).fill_(max_T)
            else:
                encoder_out_lengths = torch.sum(~padding_mask, dim=-1)
            # encoder out dim: B x T x C -> T x B x C
            encoder_out = encoder_out.transpose(0, 1)


        elif self.encoder_type == "data2vec_v2" or self.encoder_type == "hubert_v2":
            x, encoder_out_lengths = self.subsample(src_tokens, src_lengths=src_lengths)
            x = self.linear(x)
            x = self.dropout(x)
            x = x.transpose(0, 1)

            encoder_padding_mask = lengths_to_padding_mask(encoder_out_lengths)
            encoder_out, _ = self.encoder.encoder(x, padding_mask=encoder_padding_mask)
            encoder_out = encoder_out.transpose(0, 1)

        elif self.encoder_type in ["hubert_feadaptor", "data2vec_feadaptor", "wav2vec2_feadaptor"]:
            # B x T x 1
            source = src_tokens.squeeze(-1)

            ##### extract rawwav freature
            # postprocessing, normalize for data2vec
            src_padding_mask = lengths_to_padding_mask(src_lengths)
            with torch.no_grad():
                # only data2vec applied the layer norm on raw wav
                if self.encoder_type == "data2vec_feadaptor":
                    raw_source = nn.functional.layer_norm(source, source.shape[1:])
                else:
                    raw_source = source
                rawwav_features, rawwav_encoder_out_lengths = self.encoder.extract_rawwav_features(raw_source, src_padding_mask)

            ##### extract fbank feature
            x, encoder_out_lengths = self.extract_fbank_features(source, src_lengths)
            # fbank_fetures: T * B * C -> B * T * C
            fbank_features = x.transpose(0, 1)

            l2loss = self.ComputeFrontEndMSELoss(fbank_features, encoder_out_lengths, rawwav_features, rawwav_encoder_out_lengths)

            if self._step % 100 == 0:
                print (f"\t\txiexie0, step: {self._step}: l2loss: {l2loss}")
            self._step += 1


            if self._step < 20000:
            # if self._step < -1:
                x = x.detach()
            x = x.transpose(0, 1)

            encoder_padding_mask = lengths_to_padding_mask(encoder_out_lengths)
            encoder_out, _ = self.encoder.encoder(x, padding_mask=encoder_padding_mask)
            encoder_out = encoder_out.transpose(0, 1)

            # output classify layer
            outs = self.outlayer(encoder_out)

            return outs, encoder_out_lengths, l2loss


        else:
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
