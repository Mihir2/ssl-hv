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
from lightly.transforms import utils
from lightly.data import LightlyDataset

from lightly.transforms.vicreg_transform import VICRegTransform

resize = 224
# We disable resizing and gaussian blur for cifar10.
pretrain_transform = VICRegTransform(input_size=resize, gaussian_blur=0.0)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize((resize, resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)

from lightly.loss.vicreg_loss import VICRegLoss

## The projection head is the same as the Barlow Twins one
from lightly.models.modules.heads import VICRegProjectionHead
from lightly.transforms.vicreg_transform import VICRegTransform

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

class VICReg(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.results_df = pd.DataFrame(columns=['Epoch', 'val_intra', 'val_inter', 'val_intra_tsne', 'val_inter_tsne'])
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = VICRegProjectionHead(
            input_dim=512,
            hidden_dim=2048,
            output_dim=2048,
            num_layers=2,
        )
        self.criterion = VICRegLoss()
        self.epoch = 0
        self.train_loss = [0]

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

    def on_training_epoch_start(self) -> None:
        super().on_training_epoch_start()
        self.train_loss = []
        return
    
    def training_step(self, batch, batch_index):
        (x0, x1) = batch[0]
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.train_loss.append(loss.item())
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=1e-2)
        return optim

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.mean_inter_writer_distance = []
        self.mean_intra_writer_distance = []
        self.mean_tsne_inter_writer_distance = []
        self.mean_tsne_intra_writer_distance = []
        return

    def validation_step(self, batch, batch_idx):
        embeddings = self.backbone(batch[0]).flatten(start_dim=1)
        labels = batch[1]
        
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
        tsne_embeddings = tsne.fit_transform(embeddings.cpu().detach().numpy())

        mean_intra_similarity, mean_inter_similarity = mean_cosine_similarity(torch.tensor(embeddings), labels)
        mean_intra_similarity_tsne, mean_inter_similarity_tsne = mean_cosine_similarity(torch.tensor(tsne_embeddings), labels)

        self.mean_tsne_inter_writer_distance.append(mean_inter_similarity_tsne)
        self.mean_tsne_intra_writer_distance.append(mean_intra_similarity_tsne)
        
        self.mean_inter_writer_distance.append(mean_inter_similarity)
        self.mean_intra_writer_distance.append(mean_intra_similarity)

    def on_validation_epoch_end(self):
        mean_intra_writer_distance = np.array(self.mean_intra_writer_distance).mean()
        mean_inter_writer_distance = np.array(self.mean_inter_writer_distance).mean()
        
        mean_intra_tsne_writer_distance = np.array(self.mean_tsne_intra_writer_distance).mean()
        mean_inter_tsne_writer_distance = np.array(self.mean_tsne_inter_writer_distance).mean()
        
        self.log('val_intra', mean_intra_writer_distance)
        self.log('val_inter', mean_inter_writer_distance)
        self.log('val_intra_inter_diff', mean_inter_writer_distance - mean_intra_writer_distance)
        self.log('val_intra_tsne', mean_intra_tsne_writer_distance)
        self.log('val_inter_tsne', mean_inter_tsne_writer_distance)
        self.log('val_intra_inter_diff_tsne', mean_inter_writer_distance - mean_intra_tsne_writer_distance)
        
        self.epoch += 1
        self.results_df = self.results_df.append({
            'Epoch': self.epoch,
            'Train Loss': np.array(self.train_loss).mean(),
            # 'Val Loss': val_loss,
            'val_inter_tsne': mean_inter_tsne_writer_distance,
            'val_intra_tsne': mean_intra_tsne_writer_distance,
            'val_inter': mean_inter_writer_distance,
            'val_intra': mean_intra_writer_distance
        }, ignore_index=True)
        display_results(self.results_df)

model = VICReg()
