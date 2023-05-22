"""
Preprocessing script
This python module implements the preprocessing steps for generating clusters
"""

import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from utils import load_config

config_f=load_config('config.yaml')
#os.path.join(config_f["data_directory"]+config_f["clean_data"],"complete.csv")
# Define a dummy function to apply to categorical data
def identity_function(x):
        return x

def preprocessing(data):
    """
    Loads csv containing data to process in order to create clusters, then defines a preprocessor for 
    each type of variable (numerical or categorical), then stores the preprocessing steps in a pickle file
    and returns the preprocessed data. 
    Args:
    path (string): path to the input data
    Returns:
    data_preprocessed (numpy matrix): Contains the data in its preprocessed form ready to cluster 
    segmentacion (pandas dataframe): Dataframe in its original structure
    """
    #Load data
    datos = data\
            .assign(
                themes = lambda df_: df_.themes\
                    .str.replace("\[","", regex=True)\
                    .str.replace("\]","", regex=True)\
                    .str.replace("\'","", regex=True)\
                    .str.replace(",'","", regex=True),
                genres = lambda df_: df_.genres\
                    .str.replace("\[","", regex=True)\
                    .str.replace("\]","", regex=True)\
                    .str.replace("\'","", regex=True)\
                    .str.replace(",'","", regex=True)
            )
    # Calcula la frecuencia de cada estudio
    studio_counts = datos['studio'].value_counts()
    # Crea un conjunto de los estudios que aparecen 2 veces o menos
    rare_studios = set(studio_counts[studio_counts <= 30].index)
    datos['studio'] = datos['studio'].apply(lambda x: 'otros_estudios' if x in rare_studios else x)
    datos['genres'] = datos['genres'].str.split(',').apply(lambda x: [i.strip() for i in x])
    
    # Ahora convertimos las listas de géneros nuevamente a cadenas separadas por comas
    datos['genres'] = datos['genres'].apply(lambda x: ','.join(x))
    # Ahora, utiliza get_dummies para codificar las columnas
    genres_encoded = datos['genres'].str.get_dummies(',')
    sums = genres_encoded.sum()
    # Encontrar las columnas que no cumplen con tu criterio
    columns_to_drop = sums[sums < 115].index
    # Eliminar esas columnas del DataFrame
    genres_encoded = genres_encoded.drop(columns=columns_to_drop)
    segmentacion = pd.concat([datos[['score','studio','number_episodes', 'demographics','emission_type','month','members']], genres_encoded], axis=1)
    numerical_variables = config_f["clustering"]["numerical_variables"]
    categorical_variables = config_f["clustering"]["categorical_variables"]
    #Definir index de las variables categoricas
    cat_column = [1,3,4,5]
    segmentacion_kprototypes=segmentacion[config_f["clustering"]["segmentation_variables"]]
    ## Define transformers for each type of variable
    categorical_transformer = Pipeline(
            steps=[("identity", FunctionTransformer(identity_function))]
            )
    numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
            )
    # Complete preprocessing steps
    preprocessor = ColumnTransformer(
            transformers=[
                # I applied the standard scaler only to numerical variables
                ("num", numeric_transformer, numerical_variables),
                # this pipeline is flexible so more categorical variables may be added
                ("cat", categorical_transformer, categorical_variables)
                ],
            remainder='passthrough'
            )
    
    
    # save the processor to disk
    filename_process = config_f["clustering"]["preprocessor_name"]
    #Write the scaler to folder
    with open(("../"+config_f["modelos"])+filename_process, 'wb') as file_process:
        pickle.dump(preprocessor, file_process)
    #### Apply preprocessing on train data and save preprocessing params in picklle file
    preprocessor.fit(segmentacion_kprototypes)
    # Transform the training dataset
    data_preprocessed = preprocessor.transform(segmentacion_kprototypes)
    return data_preprocessed, segmentacion
