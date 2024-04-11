import random
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from scipy.signal import butter, lfilter
import librosa.feature


def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut=0.5, highcut=30, fs=200, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data).astype(np.float32)
    return y


def normalize(x):
    mean = x.mean(axis=-1, keepdims=True)
    # std = x.std(axis=-1, keepdims=True) + 1e-6
    # x = (x - mean) / std
    return x - mean


def convert_to_dual(x):
    # x: (19, 10000)
    x1 = np.concatenate([x[:4], x[8:12], x[16:18]], axis=0)
    x2 = np.concatenate([x[4:8], x[12:16], x[16:18]], axis=0)
    return np.stack([x1, x2], axis=0)


def cutout_1d(x, max_length=128, num_cutouts=4):
    num_steps = x.shape[-2]
    for i in range(4):
        for _ in range(num_cutouts):
            length = np.random.randint(0, max_length + 1)
            length = min(length, num_steps)
            start = np.random.randint(0, num_steps - length + 1)
            end = start + length
            x[:, i, start:end] = 0
    return x


def extract_eeg_feat(eeg):
    pairs = [
        ["Fp1", "F7"],
        ["F7", "T3"],
        ["T3", "T5"],
        ["T5", "O1"],
        ["Fp2", "F8"],
        ["F8", "T4"],
        ["T4", "T6"],
        ["T6", "O2"],
        ["Fp1", "F3"],
        ["F3", "C3"],
        ["C3", "P3"],
        ["P3", "O1"],
        ["Fp2", "F4"],
        ["F4", "C4"],
        ["C4", "P4"],
        ["P4", "O2"],
    ]

    eef_feat = [eeg[pair[0]] - eeg[pair[1]] for pair in pairs]
    eef_feat = np.array(eef_feat)
    return eef_feat


class MyDataset(Dataset):
    def __init__(self, cfg, npz_paths, mode):
        assert mode in ["train", "val"]
        self.cfg = cfg
        self.mode = mode
        self.transforms = get_train_transforms(cfg) if mode == "train" else get_val_transforms(cfg)
        self.npz_paths = npz_paths

        if mode == "train" and cfg.task.pseudo_prefix:
            data_root = Path(__file__).parents[1].joinpath("input")
            pseudo_npz_path = data_root.joinpath(f"{cfg.task.pseudo_prefix}_fold{self.cfg.data.fold_id}.npz")
            d = np.load(pseudo_npz_path)
            eeg_ids = d["eeg_ids"]
            eeg_sub_ids = d["eeg_sub_ids"]
            self.preds = torch.from_numpy(d["preds"])
            self.preds = torch.softmax(self.preds, dim=-1).numpy()
            self.eeg_id_to_idx = {(eeg_id, eeg_sub_id): i for i, (eeg_id, eeg_sub_id) in
                                  enumerate(zip(eeg_ids, eeg_sub_ids))}

    def __len__(self):
        return len(self.npz_paths)

    @staticmethod
    def get_eeg_from_id(eeg_id):
        eeg_path = Path(__file__).parents[1].joinpath("input", "train_eegs", f"{eeg_id}.parquet")
        eeg = pd.read_parquet(eeg_path)
        eeg.fillna(eeg.mean(), inplace=True)
        eeg_sub_offset = np.random.randint(0, len(eeg) - 10000 + 1)
        eeg_sub = eeg.iloc[eeg_sub_offset:eeg_sub_offset + 10000]
        eeg_sub_feat = extract_eeg_feat(eeg_sub)
        return eeg_sub_feat

    def __getitem__(self, idx):
        if self.mode == "train":
            npz_path = random.choice(self.npz_paths[idx])
        else:
            npz_path = self.npz_paths[idx][0]

        eeg_id = int(npz_path.stem.split("_")[0])
        eeg_sub_id = int(npz_path.stem.split("_")[1])
        npz_data = np.load(npz_path)

        if self.cfg.model.arch == "2d":
            eeg_spec = npz_data["x"]  # (19, 128, 157) type, dim, time
            eeg_spec = eeg_spec[:16]
            eeg_spec = (eeg_spec + 40) / 40
            eeg_spec = eeg_spec.reshape(4, 4, 128, 157)

            if self.mode == "train":
                eeg_spec = eeg_spec[np.random.permutation(4)]

                if np.random.rand() < 0.5:
                    eeg_spec = eeg_spec[:, :, :, ::-1].copy()

            eeg_spec = eeg_spec.transpose(0, 2, 1, 3)  # 4, 128, 4, 157
            eeg_spec = eeg_spec.reshape(4 * 128, 4 * 157)
            eeg_spec = cv2.resize(eeg_spec, (512, 256)).T

            spec = npz_data["y"]  # (400, 300)
            spec = (spec + 40) / 40

            if self.mode == "train":
                spec = spec.reshape(4, 100, 300)
                spec = spec[np.random.permutation(4)]
                spec = spec.reshape(400, 300)

                if np.random.rand() < 0.5:
                    spec = spec[:, ::-1].copy()

            spec = spec[:, 100:-100]
            spec = cv2.resize(spec.T, (256, 512))
            x = np.concatenate([eeg_spec, spec], axis=1)

            if self.mode == "train":
                if self.cfg.task.augment:
                    x = self.transforms(image=x)["image"]
                else:
                    x = x[None, ...]
            else:
                x = self.transforms(image=x)["image"]
        elif self.cfg.model.arch == "effnet1d":
            x = npz_data["z"]

            if self.mode == "train" and self.cfg.task.eeg_shift:
                x = self.get_eeg_from_id(eeg_id)

            x = butter_bandpass_filter(x, lowcut=0.5, highcut=20)
            x = x.astype(np.float32)
            x = x[:16]

            if self.cfg.task.random_downsample and self.mode == "train":
                downsample_rate = np.random.randint(4, 7)
                x = x[:, np.random.randint(downsample_rate)::downsample_rate]

                if x.shape[-1] < 1920:
                    pad_needed = 1920 - x.shape[-1]
                    pad_offset = np.random.randint(pad_needed + 1)
                    x = np.pad(x, ((0, 0), (pad_offset, pad_needed - pad_offset)), mode="constant")
                else:
                    offset = np.random.randint(x.shape[-1] - 1920 + 1)
                    x = x[:, offset:offset + 1920]
            else:
                offset = np.random.randint(400) if self.mode == "train" else 200
                x = x[:, offset:offset + 9600]
                x = x[:, ::self.cfg.task.downsample_rate]

            if self.mode == "train" and np.random.randint(2):
                x = x[:, ::-1].copy()

            x = x.reshape(4, 4, -1)
            x = x.transpose(1, 0, 2)
            x = normalize(x)

            if self.mode == "train" and np.random.randint(2):
                x = cutout_1d(x)

            if self.mode == "train":
                x = x[:, np.random.permutation(4)]

            if self.cfg.task.pretrain:
                y = librosa.feature.melspectrogram(y=x, sr=40, hop_length=1920 // 28, n_fft=256, n_mels=96,
                                                   fmin=0, fmax=20)
                y = librosa.power_to_db(y, ref=np.max)
                y = y.transpose(0, 2, 1, 3)
                y = y.reshape(-1, 4, 29)
                y = (y + 40) / 40
                return x, y, (eeg_id, eeg_sub_id)

        elif self.cfg.model.arch == "eeg2d":
            x = npz_data["z"]
            x = x[:16]
            x = butter_bandpass_filter(x)
            x = x.astype(np.float32)
            x = normalize(x) / 10.0
            x = x.reshape(16, 1000, 10)
            x = x.transpose(0, 2, 1)
            x = x.reshape(1, 160, 1000)
        else:
            raise ValueError(f"unknown model arch {self.cfg.model.arch}")

        y = npz_data["votes"]

        if self.mode == "train" and self.cfg.task.pseudo_prefix:
            if y.sum() <= 7:
                k = (eeg_id, eeg_sub_id)
                pseudo_idx = self.eeg_id_to_idx[k]
                y = self.preds[pseudo_idx]

        y = y / y.sum()

        return x, y, (eeg_id, eeg_sub_id)


def get_train_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            A.XYMasking(num_masks_x=(1, 4), num_masks_y=(1, 4), mask_y_length=(0, 32), mask_x_length=(0, 32),
                        fill_value=-1.0, p=0.5),
            # A.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT, scale_limit=0.2, value=0, rotate_limit=180,
            #                   mask_value=0),
            # A.RandomScale(scale_limit=(0.8, 1.2), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.RandomCrop(height=self.cfg.data.train_img_h, width=self.cfg.data.train_img_w, p=1.0),
            # A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.5),
            # A.RandomRotate90(p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            # A.HueSaturationValue(p=0.5),
            # A.ToGray(p=0.3),
            # A.GaussNoise(var_limit=(0.0, 0.05), p=0.5),
            # A.GaussianBlur(p=0.5),
            # A.Normalize(p=1.0, mean=23165, std=2747),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_val_transforms(cfg):
    return A.Compose(
        [
            A.Resize(height=cfg.task.img_size, width=cfg.task.img_size, p=1),
            # A.RandomScale(scale_limit=(1.0, 1.0), p=1),
            # A.PadIfNeeded(min_height=cfg.task.img_size, min_width=cfg.task.img_size, p=1.0,
            #              border_mode=cv2.BORDER_CONSTANT, value=0),
            # A.Crop(y_max=self.cfg.data.val_img_h, x_max=self.cfg.data.val_img_w, p=1.0),
            # A.Normalize(p=1.0, mean=23165, std=2747),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )
