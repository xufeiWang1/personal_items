# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform,
)

from fairseq.data.audio.emotion_recognition_dataset import (
    EmotionRecognitionDataset,
    EmotionRecognitionDatasetCreator,
)
from fairseq.dataclass.configs import FairseqConfig
from fairseq.tasks import FairseqTask, LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)

@dataclass
class EmotionRecognitionConfig(FairseqConfig):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    config_yaml: str = field(default="config.yaml", metadata={"help": "config yaml file"})
    max_source_positions: int = 6000
    max_target_positions: int = 1024
    seed: int = 0

@register_task("emotion_recognition", dataclass=EmotionRecognitionConfig)
class EmotionRecognitionTask(FairseqTask):
    cfg: EmotionRecognitionConfig

    def __init__(self, cfg, tgt_dict, blank_symbol="<blank>"):
        super().__init__(cfg)
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        self.speaker_to_id = self._get_speaker_to_id()
        self.blank_symbol = blank_symbol

    def _get_speaker_to_id(self):
        speaker_to_id = None
        speaker_set_filename = self.data_cfg.config.get("speaker_set_filename")
        if speaker_set_filename is not None:
            speaker_set_path = Path(self.cfg.data) / speaker_set_filename
            with open(speaker_set_path) as f:
                speaker_to_id = {r.strip(): i for i, r in enumerate(f)}
        return speaker_to_id

    @classmethod
    def setup_task(cls, cfg: EmotionRecognitionConfig, **kwargs):
        data_cfg = S2TDataConfig(Path(cfg.data) / cfg.config_yaml)
        dict_path = Path(cfg.data) / data_cfg.vocab_filename
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dict not found: {dict_path.as_posix()}")
        tgt_dict = Dictionary.load(dict_path.as_posix())
        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(cfg, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in cfg.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(cfg, tgt_dict)

    def build_criterion(self, criterion_cfg):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and criterion_cfg.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(criterion_cfg, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.cfg)
        bpe_tokenizer = self.build_bpe(self.cfg)
        self.datasets[split] = EmotionRecognitionDatasetCreator.from_tsv(
            self.cfg.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.cfg.seed,
            speaker_to_id=self.speaker_to_id,
        )


    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.cfg.max_source_positions, self.cfg.max_target_positions

    def build_model(self, model_cfg, from_checkpoint=False):
        model_cfg.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        model_cfg.input_channels = self.data_cfg.input_channels
        model_cfg.speaker_to_id = self.speaker_to_id
        return super(EmotionRecognitionTask, self).build_model(model_cfg, from_checkpoint)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):

        # transducer model
        if args.criterion_name == "transducer_loss":
            from fairseq.infer.transducer_beam_search_decoder import TransducerBeamSearchDecoder
            from fairseq.infer.transducer_greedy_decoder import TransducerGreedyDecoder
            extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
            if getattr(args, "print_alignment", False):
                extra_gen_cls_kwargs["print_alignment"] = True

            if seq_gen_cls is None:
                if getattr(args, "beam", 1) == 1:
                    seq_gen_cls = TransducerGreedyDecoder
                else:
                    seq_gen_cls = TransducerBeamSearchDecoder

            return seq_gen_cls(
                models,
                self.target_dictionary,
                temperature=getattr(args, "temperature", 1.0),
                # the arguments below are not being used in :class:`~TransducerGreedyDecoder`
                beam_size=getattr(args, "beam", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                max_num_expansions_per_step=getattr(
                    args, "transducer_max_num_expansions_per_step", 2
                ),
                expansion_beta=getattr(args, "transducer_expansion_beta", 0),
                expansion_gamma=getattr(args, "transducer_expansion_gamma", None),
                prefix_alpha=getattr(args, "transducer_prefix_alpha", None),
                **extra_gen_cls_kwargs,
            )


        # ctc model
        elif args.criterion_name == "ctc_loss":
            # from examples.speech_recognition.new.decoders.decoder import Decoder
            from fairseq.infer.decoders.decoder import Decoder
            return Decoder(args, self.target_dictionary)

        # seq2seq model
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if EmotionRecognitionDataset.is_lang_tag(s)
        }

        if extra_gen_cls_kwargs is None:
            extra_gen_cls_kwargs = {}
        extra_gen_cls_kwargs["symbols_to_strip_from_output"] = lang_token_ids
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return EmotionRecognitionDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
