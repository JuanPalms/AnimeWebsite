"""
Clustering script
This python module implements the clustering algortihm Kprototypes it calls the preprocessing module and 
perfoms clustering with preprocessed data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.model_selection import train_test_split
import pickle
import logging
from utils import load_config

config_f = load_config('config.yaml')

def clustering(datos_procesados, original):
    config_f = load_config('config.yaml')

    kprototypes = KPrototypes(
    n_clusters=config_f["clustering"]["number_clusters"],
    init='Cao',
    n_init=config_f["clustering"]["n_init"],
    verbose=0,
    random_state=123,
    n_jobs=config_f["clustering"]["usable_cores"])
    
    cat_column = [3,4,5,6,7,8,9,10,11,12,13,14,15]

    
    cluster_assignments = kprototypes.fit_predict(datos_procesados, categorical=cat_column)
    original['Cluster'] = cluster_assignments
    original.to_csv(os.path.join("../"+config_f["data"]["clusters"],"cluster_results.csv"))

    return original, datos_procesados, cluster_assignments
