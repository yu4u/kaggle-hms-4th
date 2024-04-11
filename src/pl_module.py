from pathlib import Path
import numpy as np
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2

from .model import get_model_from_cfg
from .loss import get_loss
from .util import mixup, get_augment_policy


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.preds = None
        self.gts = None
        self.eeg_ids = None
        self.cfg = cfg
        self.model = get_model_from_cfg(cfg, cfg.model.resume_path)

        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV2(self.model, decay=0.998)

        self.loss = get_loss(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        augment_policy = get_augment_policy(self.cfg)

        if augment_policy == "mixup":
            x, y = mixup(x, y)
        elif augment_policy == "nothing":
            pass
        else:
            raise ValueError(f"unknown augment policy {augment_policy}")

        output = self.model(x)
        loss_dict = {k: v if k == "loss" else v.detach() for k, v in self.loss(output, y).items()}
        self.log_dict(loss_dict, on_epoch=True, sync_dist=True)
        return loss_dict

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model(x)

        loss_dict = self.loss(output, y)
        log_dict = {"val_" + k: v for k, v in loss_dict.items()}
        self.log_dict(log_dict, on_epoch=True, sync_dist=True)

    def on_validation_epoch_end(self):
        pass

    def on_test_start(self):
        self.eeg_ids = []
        self.eeg_sub_ids = []
        self.gts = []
        self.preds = []

    def test_step(self, batch, batch_idx):
        x, y, (eeg_ids, eeg_sub_ids) = batch

        if self.cfg.model.ema:
            output = self.model_ema.module(x)
        else:
            output = self.model.tta(x)

        self.eeg_ids.append(eeg_ids.cpu().numpy())
        self.eeg_sub_ids.append(eeg_sub_ids.cpu().numpy())
        self.gts.append(y.detach().cpu().numpy())
        self.preds.append(output.detach().cpu().numpy())

    def on_test_epoch_end(self):
        eeg_ids = np.concatenate(self.eeg_ids)
        eeg_sub_ids = np.concatenate(self.eeg_sub_ids)
        gts = np.concatenate(self.gts)
        preds = np.concatenate(self.preds)
        data_root = Path(__file__).parents[1].joinpath("input")
        filename = Path(self.cfg.model.resume_path).stem
        output_path = data_root.joinpath(f"result__{filename}.npz")
        np.savez(output_path, eeg_ids=eeg_ids, eeg_sub_ids=eeg_sub_ids, gts=gts, preds=preds)

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)
        scheduler, num_epochs = create_scheduler_v2(optimizer=optimizer, num_epochs=self.cfg.trainer.max_epochs,
                                                    warmup_lr=self.cfg.opt.lr / 10.0, **self.cfg.scheduler)
        lr_dict = dict(
            scheduler=scheduler,
            interval="epoch",  # same as default
            frequency=1,  # same as default
        )
        return dict(optimizer=optimizer, lr_scheduler=lr_dict)

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
