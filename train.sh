#!/bin/bash
for i in {0..4}
do
    python 02_train.py trainer.accelerator=gpu trainer.devices=-1 trainer.strategy=ddp data.batch_size=96 data.num_workers=5 trainer.max_epochs=64 wandb.name=2stage_32_64_fold${i} opt.lr=2e-3 opt.opt=AdamW opt.weight_decay=1e-5 loss.mixup=0.5 model.arch=effnet1d task.head=gem trainer.deterministic=False data.fold_id=${i} trainer.precision=bf16 trainer.reload_dataloaders_every_n_epochs=32
done
