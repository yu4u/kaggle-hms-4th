# HMS - Harmful Brain Activity Classification 4th Place Solution (yu4u's Part)

This is the implementation of the 4th place solution (yu4u's part) for [HMS - Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification) at Kaggle.
The overall solution is described in [this discussion](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492240).

## Requirements

- 24GB x 2 VRAM (trained on GeForce RTX 3090 x 2).

## Preparation
- Download the competition dataset from [here](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data) and put them in `input` directory.
- Install Docker/NVIDIA Container Toolkit.
- Build the Docker image and enter the Docker container:
    ```shell
    export UID=$(id -u)
    docker compose up -d
    docker compose exec dev /bin/bash
    ```
- Login to wandb:
    ```shell
    wandb login
    ```
- Preprocess the dataset:
    ```shell
    python 00_create_dataset_eeg.py
    ```


## Training

```shell
chmod u+x train.sh
./train.sh
```

Trained weights will be saved:

```shell
$ tree saved_models/
saved_models/
├── 2stage_32_64_fold0
│         └── 2stage_32_64_fold0_epoch=055_val_loss=0.2975.ckpt
├── 2stage_32_64_fold1
│         └── 2stage_32_64_fold1_epoch=057_val_loss=0.2458.ckpt
├── 2stage_32_64_fold2
│         └── 2stage_32_64_fold2_epoch=062_val_loss=0.2545.ckpt
├── 2stage_32_64_fold3
│         └── 2stage_32_64_fold3_epoch=047_val_loss=0.2554.ckpt
└── 2stage_32_64_fold4
          └── 2stage_32_64_fold4_epoch=049_val_loss=0.2680.ckpt
```

## Evaluation

```shell
chmod u+x eval.sh
./eval.sh

# create oof file
python 04_oof.py --prefix 2stage_32_64

> Validation Loss: 0.26099773524747444
```

An oof file will be saved as

```shell
input/oof_2stage_32_64.csv
```
