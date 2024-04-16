import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--prefix", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    data_root = Path(__file__).resolve().parent.joinpath("input")
    loss_func = nn.KLDivLoss(reduction="batchmean")
    df = pd.read_csv(data_root.joinpath("train.csv"))
    df["num_votes"] = df.values[:, -6:].sum(-1)
    df = df.loc[df.groupby("eeg_id")["num_votes"].idxmax()]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    oof_eeg_ids = []
    oof_eeg_sub_ids = []
    oof_preds = []
    oof_preds_with_softmax = []
    oof_gts = []
    oof_patient_ids = []

    for i, (train_idx, val_idx) in enumerate(sgkf.split(df, df["expert_consensus"], df["patient_id"])):

        val_patient_ids = set(df.iloc[val_idx]["patient_id"].values)
        result_path = sorted(data_root.glob(f"result__{args.prefix}_fold{i}*.npz"))[0]
        d = np.load(result_path)
        eeg_ids = d["eeg_ids"]
        eeg_sub_ids = d["eeg_sub_ids"]
        gts = d["gts"]
        preds = d["preds"]
        eeg_tuple_to_index = {(eeg_id, eeg_sub_id): i for i, (eeg_id, eeg_sub_id) in
                              enumerate(zip(eeg_ids, eeg_sub_ids))}

        for _, row in df.iterrows():
            patient_id = row["patient_id"]

            if patient_id not in val_patient_ids or patient_id in oof_patient_ids:
                continue

            if row["num_votes"] < 8:
                continue

            eeg_id = row["eeg_id"]
            eeg_sub_id = row["eeg_sub_id"]
            index = eeg_tuple_to_index[(eeg_id, eeg_sub_id)]
            oof_preds.append(preds[index])
            oof_preds_with_softmax.append(torch.softmax(torch.tensor(preds[index]), dim=-1).numpy())
            oof_gts.append(gts[index])
            oof_eeg_ids.append(eeg_id)
            oof_eeg_sub_ids.append(eeg_sub_id)
            oof_patient_ids.append(row["patient_id"])

    oof_preds = np.array(oof_preds)
    oof_df = pd.DataFrame({
        "eeg_id": oof_eeg_ids,
        "eeg_sub_id": oof_eeg_sub_ids,
    })
    oof_df[["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]] = oof_preds
    oof_df.to_csv(data_root.joinpath(f"oof_{args.prefix}.csv"), index=False)
    oof_preds_with_softmax = np.array(oof_preds_with_softmax)
    oof_gts = np.array(oof_gts)
    loss = loss_func(torch.tensor(oof_preds_with_softmax).log(), torch.tensor(oof_gts))
    print(f"Validation Loss: {loss}")


if __name__ == '__main__':
    main()
