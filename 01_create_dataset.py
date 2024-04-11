import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa.feature
from scipy.signal import butter, lfilter


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--stft", action="store_true")
    parser.add_argument("--dirname", type=str, default="train_npzs")
    args = parser.parse_args()
    return args


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


def main():
    args = get_args()
    data_root = Path(__file__).resolve().parent.joinpath("input")
    eeg_dir = data_root.joinpath("train_eegs")
    spectrogram_dir = data_root.joinpath("train_spectrograms")
    df = pd.read_csv(data_root.joinpath("train.csv"))
    output_dir = data_root.joinpath(args.dirname)
    output_dir.mkdir(exist_ok=True)

    for eeg_id, sub_df in tqdm(df.groupby("eeg_id")):
        assert sub_df["spectrogram_id"].nunique() == 1
        spectrogram_id = sub_df["spectrogram_id"].iloc[0]
        eeg_path = eeg_dir.joinpath(f"{eeg_id}.parquet")
        spectrogram_path = spectrogram_dir.joinpath(f"{spectrogram_id}.parquet")
        eeg = pd.read_parquet(eeg_path)
        eeg.fillna(eeg.mean(), inplace=True)
        spectrogram = pd.read_parquet(spectrogram_path).iloc[:, 1:].values

        if args.filter:
            eeg = eeg.apply(lambda x: butter_bandpass_filter(x), axis=0)

        for idx, row in sub_df.iterrows():
            eeg_sub_id = row["eeg_sub_id"]
            eeg_label_offset_seconds = row["eeg_label_offset_seconds"]
            spectrogram_label_offset_seconds = row["spectrogram_label_offset_seconds"]
            eeg_sub_offset = int(eeg_label_offset_seconds * 200)  # 200 samples per second
            spectrogram_sub_offset = int(spectrogram_label_offset_seconds / 2.0)  # 2 seconds per spectrogram
            eeg_sub = eeg.iloc[eeg_sub_offset:eeg_sub_offset + 10000]
            spectrogram_sub = spectrogram[spectrogram_sub_offset:spectrogram_sub_offset + 300]
            eeg_sub_feat = extract_eeg_feat(eeg_sub)

            if np.isnan(spectrogram_sub).any():
                mean_value = np.nanmean(spectrogram_sub)
                spectrogram_sub[np.isnan(spectrogram_sub)] = mean_value

            spectrogram_sub = librosa.power_to_db(spectrogram_sub, ref=np.max).T

            if args.stft:
                x = librosa.stft(y=eeg_sub_feat, n_fft=1024, hop_length=64, window="hann")[:, 3:103]
                x = librosa.amplitude_to_db(abs(x), ref=np.max)
            else:
                x = librosa.feature.melspectrogram(y=eeg_sub_feat, sr=200, hop_length=64, n_fft=1024, n_mels=128,
                                                   fmin=0, fmax=20)
                x = librosa.power_to_db(x, ref=np.max)

            votes = row[["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]]
            votes = votes.values.astype(int)
            output_path = output_dir.joinpath(f"{eeg_id}_{eeg_sub_id}.npz")
            np.savez(output_path, x=x, y=spectrogram_sub, z=eeg_sub_feat, votes=votes)


if __name__ == '__main__':
    main()
