from argparse import ArgumentParser
import os.path as osp
import random

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from par.implementations.baseline.baseline import Baseline


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def main(hparams):
    model = Baseline(hparams)

    checkpoint_callback = ModelCheckpoint(
        filepath=osp.join(hparams.output_dir, "checkpoint"),
        save_best_only=True,
        monitor='val_mA',
        mode='max',
        prefix='Baseline'
    )

    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=None,
        default_save_path=hparams.output_dir,
        gpus=hparams.gpus,
        fast_dev_run=hparams.fast_dev_run,
        max_nb_epochs=hparams.max_nb_epochs,
        distributed_backend='ddp' if hparams.gpus > 1 else None,
        use_amp=hparams.use_16bit,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)

    parent_parser.add_argument(
        '-data_dir',
        type=str,
        help="data directory"
    )

    parent_parser.add_argument(
        '--fast_dev_run',
        action='store_true',
        help="if true run the training process with only with 1 training and "
        "1 validation batch"
    )

    parent_parser.add_argument(
        '-gpus',
        type=int,
        default=1,
        help="how many gpus"
    )

    parent_parser.add_argument(
        '-max_nb_epochs',
        type=int,
        default=150,
        help="how many epochs to train"
    )

    parent_parser.add_argument(
        '-output_dir',
        type=str,
        help="output directory"
    )

    parent_parser.add_argument(
        '--use_16bit',
        action='store_true',
        help="if true uses 16 bit precision"
    )

    parser = Baseline.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    main(hyperparams)
