from pathlib import Path
import torch
import torch.nn as nn
import timm
from torch.nn import Transformer
import torch.nn.functional as F

from .efficientnet1d import EfficientNet1d, GeMPool1d


class HMSModel1D(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kwargs = {
            "in_channels": 4,
            "hidden_dim": 64,
            "depth_multiplier": 4,
            "stem_kernel_size": self.cfg.model.stem_kernel_size,
            "stem_stride": self.cfg.model.stem_stride,
            "kernel_sizes": [3, 3, 5, 5, 3],
            "pool_sizes": [2, 2, 2, 2, 2],
            "layers": 3,
            "skip_in_block": True,
            "skip_in_layer": True,
            "drop_path_rate": cfg.model.drop_path_rate,
            "use_ds_conv": False,
            "se_after_dw_conv": False,
            "use_channel_mixer": True,
            "channel_mixer_kernel_size": 3,
            "mixer_type": "sc"
        }
        self.backbone = EfficientNet1d(**kwargs)
        d_model = self.backbone.out_channels + 360 if cfg.task.sim else self.backbone.out_channels
        if cfg.task.head == "trans":
            self.transformer = Transformer(d_model=d_model, dim_feedforward=d_model * 4, batch_first=True,
                                           norm_first=True, custom_decoder=nn.Identity(), num_encoder_layers=2,
                                           nhead=4)
            self.pos_emb = nn.Parameter(torch.randn(1, 1, 4, 1))
        elif cfg.task.head == "flatten":
            self.fc = nn.Sequential(
                nn.Linear(d_model * 4 * 3, d_model * 4),
                nn.BatchNorm1d(d_model * 4),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.model.drop_rate),
                nn.Linear(d_model * 4, 6))
        elif cfg.task.head == "none":
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.model.drop_rate),
                nn.Linear(d_model, 6))
        elif cfg.task.head == "gem":
            self.gem = GeMPool1d()
            self.fc = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.model.drop_rate),
                nn.Linear(d_model, 6))
        else:
            raise ValueError(f"unknown head {cfg.task.head}")

    @staticmethod
    def sim(x):
        # 2, 64, 4, 60
        b, dim, pos, time = x.shape
        x1 = x.reshape(b, dim, pos, 1, time)
        x2 = x.reshape(b, dim, 1, pos, time)
        sim = F.cosine_similarity(x1, x2, dim=1)  # 2, 4, 4, 62
        rows, cols = torch.triu_indices(pos, pos, 1)
        sim = sim[:, rows, cols]  # 2, 6, 60
        return sim.flatten(1, 2)  # 2, 360

    def __call__(self, x):
        x = self.backbone(x)

        if self.cfg.task.pretrain:
            return x

        if self.cfg.task.head == "trans":
            b, dim, pos, time = x.shape
            x = x + self.pos_emb  # b, dim, pos, time
            x = x.permute(0, 2, 3, 1)  # b, pos, time, dim
            x = x.reshape(b, pos * time, dim)
            x = self.transformer.encoder(x)
            x = x.mean(dim=1)
        elif self.cfg.task.head == "flatten":
            x = x.flatten(1, 2)  # 2, 64 * 4, 62
            x = F.adaptive_avg_pool1d(x, 3)
            x = x.flatten(1, 2)
        elif self.cfg.task.head == "none":
            x = x.mean(dim=2).mean(dim=2)  # 2, 64, 4, 62
        elif self.cfg.task.head == "gem":
            if self.cfg.task.sim:
                sim = self.sim(x)

            x = x.flatten(2, 3)  # 2, 64, 4 * 62
            x = self.gem(x)  # b c t
            x = x.squeeze(2)

            if self.cfg.task.sim:
                x = torch.cat([x, sim], dim=1)

        x = self.fc(x)
        return x

    def tta(self, x):
        # b, 4, 4, t
        return 0.5 * self(x) + 0.5 * self(x.flip(3))


class HMSModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        kwargs = {"model_name": cfg.model.backbone, "pretrained": True, "in_chans": 1, "num_classes": 6,
                  "drop_path_rate": cfg.model.drop_path_rate, "drop_rate": cfg.model.drop_rate}

        if cfg.model.backbone.startswith("swin"):
            kwargs["img_size"] = cfg.task.img_size

        self.backbone = timm.create_model(**kwargs)

    def __call__(self, x):
        x = self.backbone(x)
        return x


def get_model_from_cfg(cfg, resume_path=None):

    if cfg.model.arch == "2d":
        model = HMSModel(cfg)
    elif cfg.model.arch == "effnet1d":
        model = HMSModel1D(cfg)
    elif cfg.model.arch == "eeg2d":
        model = HMSModel(cfg)
    else:
        raise ValueError(f"unknown model arch {cfg.model.arch}")

    if resume_path:
        checkpoint = torch.load(str(resume_path), map_location="cpu")
        state_dict = {k[6:]: v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state_dict, strict=True)

    return model


class EnsembleModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        if Path(cfg.test.resume_path).is_dir():
            resume_paths = Path(cfg.test.resume_path).glob("*.ckpt")
        else:
            resume_paths = [Path(cfg.test.resume_path)]

        self.models = nn.ModuleList()

        for resume_path in resume_paths:
            model = get_model_from_cfg(cfg, resume_path)
            self.models.append(model)

    def __call__(self, x):
        outputs = [model(x) for model in self.models]
        x = torch.mean(torch.stack(outputs), dim=0)
        return x
