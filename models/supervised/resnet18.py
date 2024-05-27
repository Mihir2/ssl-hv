import random
import torch
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from PIL import Image
import PIL.ImageOps
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import pandas as pd
import os
import itertools
from datetime import datetime
from tqdm import tqdm
import os
import sys
import torch
import torch.nn as nn
import torchvision
import lightning
import lightly
import warnings
import torch
from collections import defaultdict
import numpy as np

def generate_writer_pairs(data_path, sample_percent):
    
    # Get SSL Train Writers
    ssl_train_writers = os.listdir(data_path)
    ssl_train_writers.sort()

    # Sample train writers for classification 
    num_sampled_writers = int(len(ssl_train_writers)*sample_percent)
    one_percent_writers = ssl_train_writers[:num_sampled_writers]
    
    print("Getting same writer pairs")
    # Get samples from one percent writers
    all_same_writer_pairs = []
    train_samples = []
    for writer_id in tqdm(one_percent_writers):
        writer_samples_path = os.path.join(data_path, writer_id)
        writers_samples = find_png_files_in_folders(writer_samples_path)
        # Get Same Writer Combinations
        combinations = list(itertools.combinations(writers_samples, 2))
        same_writer_pairs = pd.DataFrame(combinations, columns = ["sample1", "sample2"])
        all_same_writer_pairs.append(same_writer_pairs)

        train_samples.extend(writers_samples)

    all_same_writer_pairs = pd.concat(all_same_writer_pairs)
    all_same_writer_pairs['label'] = 1

    # Generate Different Writer Combinations
    train_df = pd.DataFrame({})
    train_df['sample_id'] = train_samples
    train_df['writer_id'] = train_df['sample_id'].str[:4]
    
    print("Getting different writer pairs")
    all_different_writer_pairs = []
    for idx, row in tqdm(all_same_writer_pairs.iterrows()):
        anchor_sample = row['sample1']
        anchor_writer = row['sample1'][:4]
        cond = train_df['writer_id'] != anchor_writer
        different_writer_sample = train_df[cond]['sample_id'].sample(1).values[0]
        different_writer_pairs = pd.DataFrame({"sample1": [anchor_sample], 
                                               "sample2": [different_writer_sample]})
        all_different_writer_pairs.append(different_writer_pairs)

    all_different_writer_pairs = pd.concat(all_different_writer_pairs)
    all_different_writer_pairs['label'] = 0

    # Join Same and Different Writer Pairs
    train_pairs = pd.concat([all_same_writer_pairs, all_different_writer_pairs])

    return train_pairs

def get_verification_dataset(generate_verification_dataset):

    if generate_verification_dataset == True:
        print("Generating the train and test path")
        print()

        ssl_train_path = "data/AND/lightly-transformed-64/supervised/train/"
        ssl_test_path = "data/AND/lightly-transformed-64/supervised/test/"

        train_sample_percent = 0.1
        test_sample_percent = 1

        print("-->Getting train writer pairs")
        train_df = generate_writer_pairs(ssl_train_path, train_sample_percent)
        print()
        print("-->Getting test writer pairs")
        test_df  = generate_writer_pairs(ssl_test_path, test_sample_percent)

        print()
        print("Number of Train Datapoints", len(train_df))
        print("Number of Test Datapoints", len(test_df))

        print()
        formatted_date = datetime.now().strftime("%m-%d-%Y-%H%M%S")
        print("Saving training and testing dataset")
        save_path = "data/AND/lightly-transformed-64/supervised/verification-dataset"
        save_path = os.path.join(save_path, formatted_date)
        os.makedirs(save_path, exist_ok=True)

        train_verification_path = os.path.join(save_path, "train.parquet")
        test_verification_path = os.path.join(save_path, "test.parquet")

        train_df.to_parquet(train_verification_path, index=False)
        test_df.to_parquet(test_verification_path, index=False)
        print("Saved Train and Test to", save_path)
    else:
        print("Setting the train and test verification path")
        save_path = "data/AND/lightly-transformed-64/supervised/verification-dataset/05-06-2024-040222" #05-06-2024-040222 #05-04-2024-221654
        train_verification_path = os.path.join(save_path, "train.parquet")
        test_verification_path = os.path.join(save_path, "test.parquet")
        
    return train_verification_path, test_verification_path


class SiameseNetworkDataset(Dataset):

    def __init__(self,transform=None,should_invert=None, is_train=True):
        self.transform = transform
        self.should_invert = should_invert        
        # Get train and test verification dataset path
        generate_verification_dataset = False
        train_path, test_path = get_verification_dataset(generate_verification_dataset)
        
        if is_train == True:
            self.img_pairs = pd.read_parquet(train_path)
        else:
             self.img_pairs = pd.read_parquet(test_path)
        self.img_pairs = self.img_pairs.sample(len(self.img_pairs))

    def __getitem__(self, index):
        
        images_path = "data/AND/lightly-transformed-64/unsupervised/"
    
        # Get Image Pair and Label
        pair = self.img_pairs.iloc[index]
        img0 = os.path.join(images_path, pair['sample1'])
        img1 = os.path.join(images_path, pair['sample2'])
        label = np.array(pair['label'], dtype=np.float32)
        
        img0 = Image.open(img0)
        img1 = Image.open(img1)
        # img0 = img0.convert("L")
        # img1 = img1.convert("L")
        
        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(label)

    def __len__(self):
        return len(self.img_pairs)
    
    def invert(sample):
    return 1 - sample

from lightly.transforms import utils

should_invert = False
resize = 64
transform = torchvision.transforms.Compose(
    [
        # T.Resize(resize),
        T.ToTensor(),
        invert
        # T.Normalize(
        #     mean=utils.IMAGENET_NORMALIZE["mean"],
        #     std=utils.IMAGENET_NORMALIZE["std"],
        # ),
    ]
)
siamese_dataset_train = SiameseNetworkDataset(transform=transform,
                                              should_invert=should_invert, 
                                              is_train=True)
siamese_dataset_test = SiameseNetworkDataset(transform=transform,
                                             should_invert=should_invert,
                                             is_train=False)

train_batch_size = 256
test_batch_size = 256
train_dataloader = DataLoader(siamese_dataset_train, batch_size=train_batch_size)
test_dataloader  = DataLoader(siamese_dataset_test, batch_size=test_batch_size)

from lightly.models.modules import AIMPredictionHead, MaskedCausalVisionTransformer
from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
import torch
from sklearn import metrics
import torchvision.models as models

class SiameseNetwork(pl.LightningModule):
    def __init__(self, learning_rate= 1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        # self.criterion = ContrastiveLoss(margin=margin)
        self.epoch = 0
        self.results_df = pd.DataFrame(columns=['Epoch', 'Train Loss', 'Val Loss', 'Val Acc', 'Val Precision', 'Val Recall', 'Val F1'])
        self.train_loss = []
        self.loss_type = "cce"
        
        # ResNet 
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Identity()
        
        # Freeze the Backbone
        deactivate_requires_grad(self.backbone)
        
        self.fc = nn.Linear(100, 64)  # Output embedding size
        
        self.cce_output_layer = nn.Sequential(
                                    nn.ReLU(),
                                    nn.Linear(64, 32),
                                    nn.ReLU(),
                                    nn.Linear(32, 2),
                                    nn.Softmax(dim=1)
                                )

    def forward_once(self, x):
        output = self.backbone.encoder(x.flatten(start_dim=1))
        output = self.fc(output[1])
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    
    def contrastive_criterion(self, output1, output2, target):
        margin = 2.0
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - target) * torch.pow(euclidean_distance, 2) +
                                      target * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
    
    def cce_criterion(self, output1, output2, target):
        l1_distance = output1 - output2 
        logits     = self.cce_output_layer(l1_distance)
        cce_loss   =  nn.CrossEntropyLoss()(logits, target.long())
        return logits, cce_loss
    
    def on_training_epoch_start(self) -> None:
        super().on_training_epoch_start()
        self.train_loss = []
        return
    
    def training_step(self, batch, batch_idx):
        x0, x1 , y = batch
        output1, output2 = self(x0, x1)
        if self.loss_type == "cce":
            logits, loss = self.cce_criterion(output1, output2, y)
        elif self.loss_type == "contrastive":
            logits, loss = self.contrastive_criterion(output1, output2, y)
        self.train_loss.append(loss.item())
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.val_loss = []
        self.val_acc = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        self.preds_list = []
        self.labels_list = []
        return
        
    def compute_similarity(self, emb1, emb2):
        similarity_scores = torch.norm(emb1 - emb2, dim=1)
        return similarity_scores
    
    def compute_accuracy(self, similarity_scores, labels, threshold=0.5):
        predictions = (similarity_scores < threshold).to(torch.float32)
        accuracy = torch.mean((predictions == labels).to(torch.float32))
        return accuracy
    
    def validation_step(self, batch, batch_idx):
        x0, x1 , y = batch
        with torch.no_grad():
            output1, output2 = self(x0, x1)
            if self.loss_type == "cce":
                logits, loss = self.cce_criterion(output1, output2, y)
                preds = logits.cpu().detach().numpy()
                labels = y.cpu().detach().numpy()

                preds_list = []
                labels_list = []
                for i in range(len(preds)):
                    if preds[i][1] > 0.5:
                        preds_list.append(1)
                    else:
                        preds_list.append(0)
                    labels_list.append(labels[i])
                val_acc = metrics.accuracy_score(preds_list, labels_list)
                val_precision = metrics.precision_score(preds_list, labels_list)
                val_recall = metrics.recall_score(preds_list, labels_list)
                val_f1 = metrics.f1_score(preds_list, labels_list)
                self.val_acc.append(val_acc)
                self.val_precision.append(val_precision)
                self.val_recall.append(val_recall)
                self.val_f1.append(val_f1)
            elif self.loss_type == "contrastive":
                logits, loss = self.contrastive_criterion(output1, output2, y)
                similarity_scores = self.compute_similarity(output1, output2)
                val_acc = self.compute_accuracy(similarity_scores, y)
                self.val_acc.append(val_acc.item())
                
            self.val_loss.append(loss.item())
            return loss
        
    def on_validation_epoch_end(self):
        train_loss = np.array(self.train_loss).mean()
        val_loss = np.array(self.val_loss).mean()
        val_acc = np.array(self.val_acc).mean()
        val_precision = np.array(self.val_precision).mean()
        val_recall = np.array(self.val_recall).mean()
        val_f1 = np.array(self.val_f1).mean()
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_precision', val_precision, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_recall', val_recall, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_f1', val_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        self.epoch += 1
        self.results_df = self.results_df.append({
            'Epoch': self.epoch,
            'Train Loss': train_loss,
            'Val Loss': val_loss,
            'Val Acc': val_acc,
            'Val Precision': val_precision,
            'Val Recall': val_recall,
            'Val F1': val_f1
        }, ignore_index=True)
        display_results(self.results_df)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
from pytorch_lightning.callbacks import EarlyStopping

early_stopping_callback = EarlyStopping(
    monitor='val_f1',
    min_delta=0.001,
    patience=10,
    mode='max',
    verbose=True
)

gpus = 1
margin = 1.0
learning_rate = 1e-3
max_epochs=200

# Setup Model Saving Directory
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
root_model_path = "models"
model_name = os.path.join("supervised-verification", "resnet-18", timestamp)
model_file_path = os.path.join(root_model_path, model_name)
print("Model CKPTS save location:", model_file_path)

model = SiameseNetwork(learning_rate=learning_rate)

trainer = pl.Trainer(max_epochs=max_epochs,
                     devices=1, 
                     accelerator="gpu",
                     default_root_dir=model_file_path,
                     log_every_n_steps=1,
                     callbacks=[early_stopping_callback])