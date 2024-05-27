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
import torchvision.transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms import MAETransform
import pytorch_lightning as pl
import torch
import torchvision
from torch import nn
from lightly.models import utils
from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.transforms import AIMTransform

min_scale = 0.2
custom_transforms = [
    T.Resize(224),
    # T.RandomResizedCrop(
    #     size=(224, 224), scale=(min_scale, 1.0), interpolation=3
    # ),  # 3 is bicubic
    # T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(
        mean=utils.IMAGENET_NORMALIZE["mean"],
        std=utils.IMAGENET_NORMALIZE["std"])
]

pretrain_transforms = MAETransform()
pretrain_transforms.transforms = T.Compose(custom_transforms)

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose(
    [
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(
            mean=utils.IMAGENET_NORMALIZE["mean"],
            std=utils.IMAGENET_NORMALIZE["std"],
        ),
    ]
)


class AIM(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        vit = MaskedCausalVisionTransformer(
            img_size=224,
            patch_size=32, 
            embed_dim=768,
            depth=12,
            num_heads=12,
            qk_norm=False,
            class_token=False,
            no_embed_class=True,
        )
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=vit.pos_embed, has_class_token=vit.has_class_token
        )
        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_patches = vit.patch_embed.num_patches

        self.backbone = vit
        self.projection_head = AIMPredictionHead(
            input_dim=vit.embed_dim, output_dim=3 * self.patch_size**2, num_blocks=1
        )

        self.criterion = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        views, targets = batch[0], batch[1]
        images = views[0]  # AIM has only a single view
        batch_size = images.shape[0]

        mask = utils.random_prefix_mask(
            size=(batch_size, self.num_patches),
            max_prefix_length=self.num_patches - 1,
            device=images.device,
        )
        features = self.backbone.forward_features(images, mask=mask)
        # Add positional embedding before head.
        features = self.backbone._pos_embed(features)
        predictions = self.projection_head(features)

        # Convert images to patches and normalize them.
        patches = utils.patchify(images, self.patch_size)
        patches = utils.normalize_mean_var(patches, dim=-1)

        loss = self.criterion(predictions, patches)
        self.log("train_metrics/train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=1.5e-4)
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

model = AIM()
