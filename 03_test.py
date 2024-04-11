import argparse
from pathlib import Path
from pytorch_lightning import Trainer
from omegaconf import OmegaConf

from src.datamodule import MyDataModule
from src.pl_module import MyModel


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, default="src/config.yaml")
    parser.add_argument("--checkpoint_prefix", type=str, default=None)
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))

    if args.checkpoint_prefix:
        cfg.model.resume_path = str(sorted(Path(f"saved_models/{args.checkpoint_prefix}_fold{cfg.data.fold_id}").glob(
            "*val_loss*.ckpt"))[0])

    model = MyModel(cfg, mode="test")
    print(OmegaConf.to_yaml(cfg))
    dm = MyDataModule(cfg)
    trainer = Trainer(**cfg.trainer)
    trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    main()
