import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import butter, lfilter


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dirname", type=str, default="train_npzs")
    args = parser.parse_args()
    return args


def butter_bandpass_filter(data, lowcut=0.5, highcut=20, fs=200, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data).astype(np.float32)
    return y


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
        ["Fz", "Cz"],
        ["Cz", "Pz"]
    ]

    eef_feat = [eeg[pair[0]] - eeg[pair[1]] for pair in pairs]
    eef_feat.append(eeg["EKG"])
    eef_feat = np.array(eef_feat)
    return eef_feat


def main():
    args = get_args()
    data_root = Path(__file__).resolve().parent.joinpath("input")
    eeg_dir = data_root.joinpath("train_eegs")
    df = pd.read_csv(data_root.joinpath("train.csv"))
    output_dir = data_root.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)

    for eeg_id, sub_df in tqdm(df.groupby("eeg_id")):
        eeg_path = eeg_dir.joinpath(f"{eeg_id}.parquet")
        eeg = pd.read_parquet(eeg_path)
        eeg.fillna(eeg.mean(), inplace=True)
        eeg = eeg.apply(lambda x: butter_bandpass_filter(x), axis=0)

        for idx, row in sub_df.iterrows():
            eeg_sub_id = row["eeg_sub_id"]
            eeg_label_offset_seconds = row["eeg_label_offset_seconds"]
            eeg_sub_offset = int(eeg_label_offset_seconds * 200)  # 200 samples per second
            eeg_sub = eeg.iloc[eeg_sub_offset:eeg_sub_offset + 10000]
            eeg_sub_feat = extract_eeg_feat(eeg_sub)

            votes = row[["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]]
            votes = votes.values.astype(int)
            output_path = output_dir.joinpath(f"{eeg_id}_{eeg_sub_id}.npz")
            np.savez(output_path, z=eeg_sub_feat, votes=votes)


if __name__ == '__main__':
    main()
