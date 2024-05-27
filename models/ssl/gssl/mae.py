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

import pytorch_lightning as pl
import torch
import torchvision
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn

from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform

class MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = vit_base_patch32_224()
        self.mask_ratio = 0.2
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=32,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        views = batch[0]
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, 32)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log("train_metrics/train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=9e-4)
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

model = MAE()