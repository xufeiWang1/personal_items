# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass

# Chinese character unit
@dataclass
class ChncharConfig(FairseqDataclass):
    sentencepiece_model: str = field(
        default="???", metadata={"help": "path to store Chinese-style character vocab file"}
    )


@register_bpe("chnchar", dataclass=ChncharConfig)
class ChncharBPE(object):
    def __init__(self, cfg):
        self.sentencepiece_model = cfg.sentencepiece_model
        self.word2id = {}
        self.word2cnt = {}
        with open(self.sentencepiece_model, 'r', encoding='utf-8') as f_voc:
            lines = f_voc.read().strip().split('\n')
            for line in lines:
                items = line.strip().split()
                if len(items) == 2:
                    word = items[0]
                    cnt  = items[1]
                    assert word not in self.word2id, f"{word} appears in voc file multiple times"
                    self.word2id[word] = len(self.word2id)
                    self.word2cnt[word] = cnt


    def encode(self, x: str) -> str:
        words = [w for w in x.strip()]
        words_tn = [w if w in self.word2id else "<unk>" for w in words]
        return ' '.join(words_tn)

    def decode(self, x: str) -> str:
        return x.replace(" ", "").strip()

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ["<unk>", "<s>", "</s>", "<pad>"]:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return False
