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
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.transforms import utils
from lightly.data import LightlyDataset

# We disable resizing and gaussian blur for cifar10.
pretrain_transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=45, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=45, gaussian_blur=0.0),
)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((45, 45)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead

class BarlowTwins(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)
        self.criterion = BarlowTwinsLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.mean_inter_writer_distance = []
        self.mean_intra_writer_distance = []
        return

    def validation_step(self, batch, batch_idx):
        embeddings = self.backbone(batch[0]).flatten(start_dim=1)
        labels = batch[1]
        mean_intra_similarity, mean_inter_similarity = mean_cosine_similarity(torch.tensor(embeddings), labels)
        self.mean_inter_writer_distance.append(mean_inter_similarity)
        self.mean_intra_writer_distance.append(mean_intra_similarity)

    def on_validation_epoch_end(self):
        mean_intra_writer_distance = np.array(self.mean_intra_writer_distance).mean()
        mean_inter_writer_distance = np.array(self.mean_inter_writer_distance).mean()
        self.log('val_intra', mean_intra_writer_distance)
        self.log('val_inter', mean_inter_writer_distance)

model = BarlowTwins()