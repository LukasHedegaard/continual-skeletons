import torch
from ride.optimizers import (
    OptimizerMixin,
    Configs,
    SgdOptimizer,
    discounted_steps_per_epoch,
)
from typing import Callable
from operator import attrgetter


class SgdMultiStepLR(OptimizerMixin):
    hparams: ...
    parameters: Callable
    train_dataloader: Callable

    def validate_attributes(self):
        attrgetter("parameters")(self)
        attrgetter("train_dataloader")(self)
        attrgetter("hparams.max_epochs")(self)
        attrgetter("hparams.batch_size")(self)
        attrgetter("hparams.num_gpus")(self)
        attrgetter("hparams.accumulate_grad_batches")(self)
        attrgetter("hparams.limit_train_batches")(self)
        for hparam in SgdMultiStepLR.configs().names:
            attrgetter(f"hparams.{hparam}")(self)

    @staticmethod
    def configs() -> Configs:
        c = SgdOptimizer.configs()
        c.add(
            name="multi_step_lr_gamma",
            type=float,
            default=0.1,
            choices=(0.001, 1),
            strategy="loguniform",
            description="Multiplicative factor for LR reduction.",
        )
        for epoch in range(1, 6):
            c.add(
                name=f"multi_step_lr_epoch{epoch}",
                type=int,
                default=-1,
                choices=(1, 10000),
                strategy="uniform",
                description="Epoch at which to reduce the learning rate.",
            )
        return c

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
        )
        steps_per_epoch = (
            self.hparams.limit_train_batches
            if self.hparams.limit_train_batches > 1
            else discounted_steps_per_epoch(
                len(self.train_dataloader()),
                self.hparams.num_gpus,
                self.hparams.accumulate_grad_batches,
            )
        )
        milestones = [
            getattr(self.hparams, f"multi_step_lr_epoch{e}") for e in range(1, 6)
        ]
        milestones = [s * steps_per_epoch for s in milestones if s > 0]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.hparams.multi_step_lr_gamma
        )
        return [optimizer], {"scheduler": scheduler, "interval": "step"}
