# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class LinearDecayLRScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    max_update: float = II("optimization.max_update")
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("linear_decay", dataclass=LinearDecayLRScheduleConfig)
class LinearDecayLRSchedule(FairseqLRScheduler):
    """Linear Decay learning rate schedulr

        - warmup stage, starting from 0, linearly
          increased to `lr` in `warmup_updates` iterations

        - decay state, starting from lr, linearly decay to 0

    During warmup::

      init_lr = 0
      lr = cfg.lr * step / cfg.warmup_updates

    During decay::

      lr = lr - (step - cfg.warmup_updates) / (cfg.max_update - cfg.warmup_updates) * lr

    """

    def __init__(self, cfg: LinearDecayLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)
        if len(cfg.lr) > 1:
            raise ValueError(
                "Cannot use a fixed learning rate schedule with tri-stage lr."
                " Consider --lr-scheduler=fixed instead."
            )

        # calculate LR at each point
        self.peak_lr = cfg.lr[0]
        self.warmup_updates = cfg.warmup_updates
        self.total_steps  = cfg.max_update

        # initial learning rate
        self.init_lr = 0
        self.lr = self.init_lr
        self.optimizer.set_lr(self.lr)


    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        super().step(epoch, val_loss)
        # we don't change the learning rate at epoch boundaries
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.warmup_updates:
            self.lr = self.peak_lr * num_updates / self.warmup_updates

        else:
            self.lr = self.peak_lr *  max(0.0, (self.total_steps - num_updates) / max(1.0, self.total_steps - self.warmup_updates))

        self.optimizer.set_lr(self.lr)

        return self.lr
