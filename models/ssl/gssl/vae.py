# This script is reused from https://github.com/williamFalcon/pytorch-lightning-vae

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
import torchvision.transforms as T
from lightly.data import LightlyDataset
from lightly.transforms import utils

import warnings
L.seed_everything(42)

class ImageSampler(L.Callback):
    def __init__(self):
        super().__init__()
        self.img_size = 64
        self.num_preds = 16

    def on_train_epoch_end(self, trainer, pl_module):
        figure(figsize=(8, 3), dpi=300)

        # Z COMES FROM NORMAL(0, 1)
        eps = 1e-6
        rand_v = torch.rand((self.num_preds, pl_module.hparams.latent_dim), device=pl_module.device)
        p = torch.distributions.Normal(torch.zeros_like(rand_v) + eps, torch.zeros_like(rand_v) + eps)
        z = p.rsample()

        # SAMPLE IMAGES
        with torch.no_grad():
            pred = pl_module.decoder(z.to(pl_module.device)).cpu()

        # UNDO DATA NORMALIZATION
        normalize = cifar10_normalization()
        mean, std = np.array(normalize.mean), np.array(normalize.std)
        img = make_grid(pred).permute(1, 2, 0).numpy() * std + mean

        # PLOT IMAGES
        trainer.logger.experiment.add_image('img',torch.tensor(img).permute(2, 0, 1), global_step=trainer.global_step)
        
sampler = ImageSampler()

size = 64
custom_transforms = [
    # T.Resize(size),
    # T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    invert,
    # T.Normalize((0.1307,0.1307,0.1307), (0.3081,0.3081,0.3081))
    # T.Normalize(
    #     mean=utils.IMAGENET_NORMALIZE["mean"],
    #     std=utils.IMAGENET_NORMALIZE["std"],
    # ),
]

transforms = T.Compose(custom_transforms)

class VAE(L.LightningModule):
    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=64):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl

    def training_step(self, batch, batch_idx):
        x = batch[0]

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)

        # kl
        kl = self.kl_divergence(z, mu, std)

        # elbo
        elbo = (kl - recon_loss)
        elbo = elbo.mean()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean()
        })

        return elbo
    
sampler = ImageSampler()
model= VAE()