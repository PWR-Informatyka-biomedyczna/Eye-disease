import argparse
from typing import NoReturn

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import random

DEFAULT_HPARAMS = {

}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    return parser.parse_args()


def seed_everything(seed: int) -> NoReturn:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main():
    seed_everything(0)
    logger = WandbLogger()


if __name__ == '__main__':
    main()
