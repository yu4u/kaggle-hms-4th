#!/bin/bash
for i in {0..4}
do
    python 03_test.py --checkpoint_prefix 2stage_32_64 trainer.accelerator=gpu trainer.devices=[0] data.batch_size=128 data.num_workers=10 model.arch=effnet1d task.head=gem data.fold_id=${i}
done
