# Bigan implementation from https://github.com/jaeho3690/BidirectionalGAN/blob/main/modules.py

import lightning as L
L.seed_everything(1234)
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)
from pl_bolts.datamodules import CIFAR10DataModule, ImagenetDataModule
from matplotlib.pyplot import imshow, figure
import numpy as np
from torchvision.utils import make_grid
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
import pytorch_lightning as pl
import torch
import os
import sys
import torch
import torch.nn as nn
import torchvision
import lightning
import lightly
import torch
from collections import defaultdict
import numpy as np
import math
import os
import time
import urllib.request
from urllib.error import HTTPError
import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import numpy as np
import seaborn as sns
import tabulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from IPython.display import HTML, display
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from matplotlib.colors import to_rgb
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm.notebook import tqdm
import warnings
import torchvision.transforms as T
from lightly.data import LightlyDataset
from lightly.transforms import utils
from argparse import ArgumentParser, Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import cli_lightning_logo
from lightning.pytorch.core import LightningModule
from lightning.pytorch.demos.mnist_datamodule import MNISTDataModule
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.utilities.imports import _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision

size = 64
custom_transforms = [
    T.ToTensor(),
    invert,
]

transforms = T.Compose(custom_transforms)


class Generator(nn.Module):
    """
    >>> Generator(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    Generator(
      (model): Sequential(...)
    )
    """

    def __init__(self, latent_dim: int = 100, img_shape: tuple = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return (img,z)
    
class Encoder(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Encoder,self).__init__()
        self.img_shape= img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            return layers

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)),1024),
            *block(1024, 512, normalize=True),
            *block(512, 256, normalize=True),
            *block(256, 128, normalize=True),
            *block(128, latent_dim, normalize=True),
            nn.Tanh()
        )

    def forward(self, img):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        z = self.model(img)
        img = img.view(img.size(0), *self.img_shape)
        
        return (img,z)

class Discriminator(nn.Module):
    def __init__(self, latent_dim,img_shape):
        super(Discriminator, self).__init__()

        joint_shape = latent_dim + np.prod(img_shape)

        self.model = nn.Sequential(
            nn.Linear(joint_shape, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, z):
        # img:[batch_size,img_size,img_size], z:[batch_size,latent_dim]
        joint = torch.cat((img.view(img.size(0),-1),z),dim=1)
        validity = self.model(joint)

        return validity


class GAN(LightningModule):
    """
    >>> GAN(img_shape=(1, 8, 8))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GAN(
      (generator): Generator(
        (model): Sequential(...)
      )
      (discriminator): Discriminator(
        (model): Sequential(...)
      )
    )
    """

    def __init__(
        self,
        img_shape: tuple = (3, 64, 64),
        lr: float = 0.0002,
        b1: float = 0.5,
        b2: float = 0.999,
        latent_dim: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # networks
        self.generator = Generator(latent_dim=self.hparams.latent_dim, img_shape=img_shape)
        self.encoder = Encoder(self.hparams.latent_dim,img_shape=img_shape)
        self.img_shape = img_shape
        self.discriminator = Discriminator(self.hparams.latent_dim, img_shape=img_shape)
        
        self.validation_z = torch.randn(8, self.hparams.latent_dim)

        self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    @staticmethod
    def adversarial_loss(y_hat, y):
        return F.binary_cross_entropy_with_logits(y_hat, y)

    def training_step(self, batch):
        imgs, _,_ = batch

        opt_g, opt_d = self.optimizers()

        # sample noise
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)
        
        # Encoder Real
        imgs = imgs.reshape(-1,np.prod(self.img_shape))
        (original_img, z_) = self.encoder(imgs)
        predict_encoder = self.discriminator(original_img, z_)
        
        # Train generator
        # ground truth result (ie: all fake)
        # put on GPU because we created this tensor inside training_loop
        valid = torch.ones(imgs.size(0), 1)
        valid = valid.type_as(imgs)
        
        fake = torch.zeros(imgs.size(0), 1)
        fake = fake.type_as(imgs)
        
        (gen_img,z)=self.generator(z)
        predict_generator = self.discriminator(gen_img,z)
        
        self.toggle_optimizer(opt_g)
        # adversarial loss is binary cross-entropy
        g_loss = (self.adversarial_loss(predict_generator, valid)+self.adversarial_loss(predict_encoder, fake)) * 0.5
        opt_g.zero_grad()
        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        # Train discriminator
        # Measure discriminator's ability to classify real from generated samples
        # how well can it label as real?
        
        z = torch.randn(imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(imgs)        
        (gen_img,z)=self.generator(z)
        (original_img,z_)= self.encoder(imgs)
        predict_encoder = self.discriminator(original_img,z_)
        predict_generator = self.discriminator(gen_img,z)

        self.toggle_optimizer(opt_d)
        d_loss = (self.adversarial_loss(predict_encoder,valid)+self.adversarial_loss(predict_generator,fake)) *0.5
        
        opt_d.zero_grad()
        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.log_dict({"d_loss": d_loss, "g_loss": g_loss})

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return opt_g, opt_d

    def on_train_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs[0])
        for logger in self.loggers:
            logger.experiment.add_image("generated_images", grid, self.current_epoch)
            
model = GAN()