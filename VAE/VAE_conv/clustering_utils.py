#!/usr/bin/env python
# coding: utf-8
# @author Souheib Ben Mabrouk


import matplotlib.pyplot as plt
from umap import UMAP  
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import numpy as np

def umap(z,latent_space ,clustering_labels ,true_labels=None ,cmap='viridis' ,aff_true_labels=True, data=None):
    # 2D representation latent space n>2 matplotlib

    umap_2d = UMAP(n_components=2, init='random', random_state=0) 
    proj_2d = umap_2d.fit_transform(z)  
    params = {'backend': 'Agg',
            "font.family": "calibri",
                } # extend as needed
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25
    fig = plt.figure()

    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"]  = 1.25

    fig, ax=plt.subplots(layout="constrained")
    scatter =ax.scatter(x=proj_2d[:,0], y=proj_2d[:,1], s=3, c=clustering_labels, alpha=0.5, cmap=cmap)
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="upper left", title="Classes",fontsize = 5)
    ax.set_facecolor("white")
    ax.add_artist(legend1)
    plt.title(f'Prediction (latent space={latent_space})',fontsize = 20)
    plt.xlabel('UMAP_1', fontsize = 24)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.ylabel('UMAP_2',fontsize = 24)
    plt.savefig('./Results/Umap_prediction_Zdim'+str(latent_space)+str(data)+'.png')
    plt.show()
    if aff_true_labels:
        fig, ax=plt.subplots(layout="constrained")
        scatter =ax.scatter(x=proj_2d[:,0], y=proj_2d[:,1], s=3, c=true_labels, alpha=0.5, cmap=cmap)
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="upper left", title="Classes", fontsize = 5)

        ax.set_facecolor("white")
        ax.add_artist(legend1)
        plt.title(f'True Label (latent space={latent_space})', fontsize=20)
        plt.xlabel('UMAP_1', fontsize = 24)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.ylabel('UMAP_2',fontsize = 24)
        plt.savefig('./Results/Umap_gt_Zdim'+str(latent_space)+str(data)+'.png')
        plt.show()
        
    
    
def tSNE(z, latent_space, clustering_labels, true_labels=None, cmap='viridis', aff_true_labels=True,data=None):
    # t-SNE projection
    tSNE_2d = TSNE(n_components=2) 
    proj_2d = tSNE_2d.fit_transform(z)  
    
    params = {'backend': 'Agg', "font.family": "calibri"}  # extend as needed
    plt.rcParams["axes.edgecolor"] = "0.15"
    plt.rcParams["axes.linewidth"] = 1.25
    fig = plt.figure()
    fig, ax = plt.subplots(layout="constrained")
    scatter = ax.scatter(x=proj_2d[:, 0], y=proj_2d[:, 1], s=3, c=clustering_labels, alpha=0.5, cmap=cmap)
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes", fontsize=5)
    
    ax.set_facecolor("white")
    ax.add_artist(legend1)
    plt.title(f'Prediction (latent space={latent_space})', fontsize=20)
    plt.xlabel('tSNE_1', fontsize=24)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel('tSNE_2', fontsize=24)
    plt.savefig('./Results/tSNE_prediction_Zdim'+str(latent_space)+str(data)+'.png')
    plt.show()

    # Display the true labels if aff_true_labels is True
    if aff_true_labels:
        fig, ax = plt.subplots(layout="constrained")
        
        scatter = ax.scatter(x=proj_2d[:, 0], y=proj_2d[:, 1], s=3, c=true_labels, alpha=0.5, cmap=cmap)
        legend1 = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes", fontsize=5)
        
        ax.set_facecolor("white")
        ax.add_artist(legend1)
        plt.title(f'True Label (latent space={latent_space})', fontsize=20)
        plt.xlabel('tSNE_1', fontsize=24)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylabel('tSNE_2', fontsize=24)
        plt.savefig('./Results/tSNE_gt_Zdim'+str(latent_space)+str(data)+'.png')
        plt.show()
        

def compute_most_represented_class_per_cluster(labels, y_test):
    nb_labels = np.max(labels)
    most_represented_classes = np.zeros(nb_labels+1)
    for label in range(nb_labels+1):
        List = list(y_test[np.where(labels == label)[0]])
        most_represented_classes[label] = max(set(List), key = List.count)
    return most_represented_classes

def substitute_classes_labels(labels, class_equivalence):
    labels_equi = labels.copy()
    nb_labels = np.max(labels)
    for label in range(nb_labels+1):
        labels_equi[np.where(labels == label)] = class_equivalence[label]
        
    return labels_equi