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
from lightly.transforms import SimCLRTransform, utils
from lightly.data import LightlyDataset

# disable blur because we're working with handwritten images
pretrain_transform = SimCLRTransform(
    input_size=45,
    gaussian_blur=0.0,
    vf_prob=0.5, 
    rr_prob=0.5
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

from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead


class SimCLRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        self.criterion = NTXentLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

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
        
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

model = SimCLRModel()