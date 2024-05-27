# Using conda_pytorch_p310
# Using ml.g5.2xlarge (1 GPU - 24GiB Memory) (8 CPUs - 32 GiB Memory)
# Nvidia (A10G)

import os
import sys
import torch
import torchvision
import lightning
import lightly
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, CategoricalColorMapper
from bokeh.palettes import Category10

output_notebook()

def count_png_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                count += 1
    return count

import torch
from collections import defaultdict
import numpy as np
def cosine_similarity(embeddings):
    """
    Compute pairwise cosine similarity between embeddings.
    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) containing embeddings.
    Returns:
        torch.Tensor: Pairwise cosine similarity matrix of shape (N, N).
    """
    embeddings /= torch.norm(embeddings, dim=1, keepdim=True)
    return torch.matmul(embeddings, embeddings.t())

def mean_cosine_similarity(embeddings, labels):
    """
    Compute the mean intra-label and inter-label cosine similarities.
    Args:
        embeddings (torch.Tensor): Tensor of shape (N, D) containing embeddings.
        labels (torch.Tensor): Tensor of shape (N,) containing labels for each embedding.
    Returns:
        float: Mean intra-label cosine similarity.
        float: Mean inter-label cosine similarity.
    """
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    
    # Group embeddings by labels
    label_to_embeddings = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_embeddings[label.item()].append(embeddings[i])

    # Calculate mean cosine similarity within each label
    intra_label_similarities = []
    for embeddings_list in label_to_embeddings.values():
        if len(embeddings_list) < 2:
            continue
        embeddings_tensor = torch.stack(embeddings_list)
        label_similarity = torch.mean(similarity_matrix[:len(embeddings_list), :len(embeddings_list)])
        intra_label_similarities.append(label_similarity.item())

    # Calculate mean cosine similarity between different labels
    inter_label_similarities = []
    for embeddings_list1 in label_to_embeddings.values():
        for embeddings_list2 in label_to_embeddings.values():
            if embeddings_list1 is embeddings_list2:
                continue
            label_similarity = torch.mean(similarity_matrix[:len(embeddings_list1), len(embeddings_list1):len(embeddings_list1)+len(embeddings_list2)])
            inter_label_similarities.append(label_similarity.item())

    return torch.tensor(intra_label_similarities).mean().item(), torch.tensor(inter_label_similarities).mean().item()


def tsne_2d(embeddings):
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init='pca')
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings

def bokeh_plot(tsne_embeddings, labels, image_data_path):

    # Sample 10 Writers and Get their samples for visualization
    tsne_embeddings_df = pd.DataFrame(tsne_embeddings, columns=["x", "y"])
    tsne_embeddings_df['sample_id'] = labels
    tsne_embeddings_df['writer_id'] = tsne_embeddings_df['sample_id'].str[:4].astype(int)
    tsne_embeddings_df['image_path'] = image_data_path + "/" + tsne_embeddings_df['sample_id']
    sampled_writers = pd.Series(tsne_embeddings_df['writer_id'].unique()).sample(10)
    tsne_embeddings_df = tsne_embeddings_df[tsne_embeddings_df['writer_id'].isin(sampled_writers)]

    source = ColumnDataSource(data={'x': tsne_embeddings_df["x"], 
                                    'y': tsne_embeddings_df["y"], 
                                    'color': tsne_embeddings_df["writer_id"].astype(str), 
                                    'label': tsne_embeddings_df["writer_id"],
                                    'image_path': tsne_embeddings_df["image_path"]})

    color_mapper = CategoricalColorMapper(factors=list(tsne_embeddings_df["writer_id"].astype(str).unique()), 
                                          palette=Category10[tsne_embeddings_df["writer_id"].nunique()])

    # Define the HTML template for hover tooltip
    hover_html = """
        <div>
            <img src="@image_path" alt="Image" style="width:200px; height:200px;">
        </div>
    """

    p = figure(title='AND dataset - T-SNE', 
               tools=[HoverTool(tooltips=hover_html)], 
               x_axis_label='T-SNE component X', 
               y_axis_label='T-SNE component Y' )


    p.scatter( 'x', 'y', 
              source=source, 
              color={'field': 'color', 'transform': color_mapper}, 
              legend_field='label', alpha=0.8, size=8 )

    return p

