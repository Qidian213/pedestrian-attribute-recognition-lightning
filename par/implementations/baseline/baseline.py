from argparse import ArgumentParser
from collections import OrderedDict
import logging
import os.path as osp

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms

from par.common import backbones
from par.common.layers.classifier import make_classifier
from par.common.dataset.dataset import Dataset
from par.common.metrics.example_based import example_based_metrics
from par.common.metrics.label_based import compute_mean_accuracy


class Baseline(LightningModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.__build_model()

        if self.hparams.weighted_loss:
            positive_ratios = np.load(
                osp.join(self.hparams.data_dir, "dataset",
                         "positive_ratios.npy"))
            self.weight_pos = torch.from_numpy(
                np.exp(1. - positive_ratios)).float()
            self.weight_neg = torch.from_numpy(
                np.exp(positive_ratios)).float()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        self.backbone, feature_size = \
            getattr(backbones, self.hparams.backbone)()

        self.classifier = make_classifier(
            self.hparams.backbone, feature_size, self.hparams.num_classes)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = F.dropout(x, self.hparams.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def criterion(self, outputs, labels):
        if self.hparams.weighted_loss:
            weight = torch.where(
                labels.cpu() == 1, self.weight_pos, self.weight_neg)

            if self.on_gpu:
                weight = weight.cuda(outputs.device.index)
        else:
            weight = None

        loss = F.binary_cross_entropy_with_logits(outputs, labels, weight)

        return loss

    def predict(self, outputs):
        if self.on_gpu:
            outputs = outputs.cuda(outputs.device.index)
        outputs = outputs.detach()
        predictions = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch

        outputs = self.forward(x)

        loss = self.criterion(outputs, y)

        predictions = self.predict(outputs)
        mA = compute_mean_accuracy(predictions, y.cpu().numpy()).mean()

        tqdm_dict = {'train_mA': mA}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self.forward(x)

        loss = self.criterion(outputs, y)

        predictions = self.predict(outputs)

        output = OrderedDict({
            'loss': loss,
            'predictions': predictions,
            'labels': y.cpu().numpy()
        })

        return output

    def validation_end(self, outputs):
        avg_loss = 0
        predictions = []
        labels = []
        for output in outputs:
            loss = output['loss']
            avg_loss += loss

            predictions.extend(output['predictions'])
            labels.extend(output['labels'])

        avg_loss /= len(outputs)

        predictions = np.array(predictions)
        labels = np.array(labels)

        mA = compute_mean_accuracy(predictions, labels).mean()

        accuracy, precision, recall, f1 = \
            example_based_metrics(predictions, labels)

        tqdm_dict = {
            'val_loss': avg_loss,
            'val_mA': mA,
            'val_acc': accuracy,
            'val_prec': precision,
            'val_recall': recall,
            'val_f1': f1,
        }

        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            'val_loss': avg_loss}

        return result

    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), self.hparams.lr,
                              momentum=self.hparams.momentum,
                              weight_decay=self.hparams.weight_decay,
                              nesterov=True)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_nb_epochs)

        return [optimizer], [scheduler]

    def __dataloader(self, train):
        normalize = [transforms.ToTensor(),
                     transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                          std=(0.229, 0.224, 0.225))]

        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((256, 128)),
                transforms.Pad(10),
                transforms.RandomCrop((256, 128)),
            ] + normalize)
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 128))
            ] + normalize)

        dataset = Dataset(root=self.hparams.data_dir,
                          split='train' if train else 'test',
                          transform=transform)

        # When using multi-node (ddp) we need to add the datasampler
        if self.use_ddp:
            sampler = DistributedSampler(dataset, shuffle=train)
        else:
            sampler = None

        loader = DataLoader(dataset=dataset,
                            batch_size=self.hparams.batch_size,
                            shuffle=train and sampler is None,
                            sampler=sampler,
                            num_workers=self.hparams.num_workers,
                            pin_memory=True,
                            worker_init_fn=_init_fn)

        return loader

    @pl.data_loader
    def train_dataloader(self):
        logging.info('training data loader called')
        return self.__dataloader(train=True)

    @pl.data_loader
    def val_dataloader(self):
        logging.info('val data loader called')
        return self.__dataloader(train=False)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('-backbone', default='resnet50', type=str)
        parser.add_argument('-batch_size', default=64, type=int)
        parser.add_argument('-dropout', default=0., type=float)
        parser.add_argument('-lr', default=0.01, type=float)
        parser.add_argument('-momentum', default=0.9, type=float)
        parser.add_argument('-num_classes', default=51, type=int)
        parser.add_argument('-num_workers', default=8, type=int)
        parser.add_argument('-weight_decay', default=0.0005, type=float)
        parser.add_argument('--weighted_loss', action='store_true')

        return parser


def _init_fn(worker_id):
    np.random.seed(0)
