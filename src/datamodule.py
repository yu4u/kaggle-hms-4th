from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from .dataset import MyDataset


class MyDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_all_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.root_dir = Path(__file__).parents[1].joinpath("input")

    def prepare_data(self):
        pass

    def get_split(self, df):
        if self.cfg.task.stratified:
            df = df.loc[df.groupby("eeg_id")["num_votes"].idxmax()]
            sgkf = StratifiedGroupKFold(n_splits=self.cfg.data.fold_num, shuffle=True, random_state=42)

            for i, (train_idx, val_idx) in enumerate(sgkf.split(df, df["expert_consensus"], df["patient_id"])):
                if i == self.cfg.data.fold_id:
                    break

            train_patient_ids = set(df.iloc[train_idx]["patient_id"].values)
            val_patient_ids = set(df.iloc[val_idx]["patient_id"].values)
        else:
            split_df = pd.read_csv(Path(__file__).parents[1].joinpath("misc/patient_fold.csv"))
            train_patient_ids = set(split_df[split_df["fold"] != self.cfg.data.fold_id]["patient_id"])
            val_patient_ids = set(split_df[split_df["fold"] == self.cfg.data.fold_id]["patient_id"])

        return train_patient_ids, val_patient_ids

    def setup(self, stage=None):
        if stage == "test":
            self.setup_test()
            return

        df = pd.read_csv(self.root_dir.joinpath("train.csv"))
        df["num_votes"] = df.values[:, -6:].sum(-1)
        train_patient_ids, val_patient_ids = self.get_split(df)
        npz_dir = self.root_dir.joinpath("train_npzs")
        train_eeg_id_to_npz_paths = defaultdict(list)
        train_all_eeg_id_to_npz_paths = defaultdict(list)
        val_eeg_id_to_npz_paths = defaultdict(list)

        for _, row in df.iterrows():
            eeg_id = row["eeg_id"]
            eeg_sub_id = row["eeg_sub_id"]
            patient_id = row["patient_id"]
            num_votes = row["num_votes"]
            npz_path = npz_dir.joinpath(f"{eeg_id}_{eeg_sub_id}.npz")
            assert npz_path.is_file()

            if patient_id in train_patient_ids:
                train_all_eeg_id_to_npz_paths[eeg_id].append(npz_path)

                if num_votes > 7:
                    train_eeg_id_to_npz_paths[eeg_id].append(npz_path)
            elif patient_id in val_patient_ids:
                if num_votes > 7:
                    val_eeg_id_to_npz_paths[eeg_id].append(npz_path)
            else:
                raise ValueError(f"unknown patient_id {patient_id}")

        self.train_all_dataset = MyDataset(self.cfg, list(train_all_eeg_id_to_npz_paths.values()), "train")
        self.train_dataset = MyDataset(self.cfg, list(train_eeg_id_to_npz_paths.values()), "train")
        self.val_dataset = MyDataset(self.cfg, list(val_eeg_id_to_npz_paths.values()), "val")
        print(f"train: {len(self.train_dataset)}, val: {len(self.val_dataset)}")

    def setup_test(self):
        df = pd.read_csv(self.root_dir.joinpath("train.csv"))
        # df["num_votes"] = df.values[:, -6:].sum(-1)
        # df = df[df["num_votes"] <= 7]
        npz_dir = self.root_dir.joinpath(self.cfg.task.dirname)
        npz_paths = []

        for _, row in df.iterrows():
            eeg_id = row["eeg_id"]
            eeg_sub_id = row["eeg_sub_id"]
            npz_path = npz_dir.joinpath(f"{eeg_id}_{eeg_sub_id}.npz")
            assert npz_path.is_file()
            npz_paths.append([npz_path])

        self.test_dataset = MyDataset(self.cfg, npz_paths, "val")

    def train_dataloader(self):
        if self.cfg.trainer.reload_dataloaders_every_n_epochs > 0 and self.trainer.current_epoch < self.cfg.trainer.reload_dataloaders_every_n_epochs - 1:
            return DataLoader(self.train_all_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                              shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)
        else:
            return DataLoader(self.train_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                              shuffle=True, drop_last=True, num_workers=self.cfg.data.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.cfg.data.batch_size, collate_fn=None,
                          shuffle=False, drop_last=False, num_workers=self.cfg.data.num_workers)
