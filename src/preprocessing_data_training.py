"""
This python module performs the data preprocessing necessary for fitting a xgboost classifier to anime data
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import GridSearchCV
import os
from utils import load_config, tabulacion, apply_format, impute_missing_drop_columns
import pickle
import plotly.graph_objects as go
import xgboost as xgb

CURRENT=os.getcwd()
ROOT=os.path.dirname(CURRENT)

config_f=load_config("config.yaml")


datos=pd.read_csv(os.path.join(ROOT,config_f["data"]["main"]))
#datos = datos[datos['studio'] != 'add_some']
#datos = datos[datos['demographics'] != 'not_available']
genres=pd.read_csv(os.path.join(ROOT,config_f["data"]["cat_genres"]))
themes=pd.read_csv(os.path.join(ROOT,config_f["data"]["cat_themes"]))


datos = datos\
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
numerical = ['number_episodes']
nominal = ['studio', 'genres', 'demographics', 'emission_type', 'year', 'month']
datos,numerical,nominal=impute_missing_drop_columns(datos,numerical,nominal)
y=datos["score"]
datos_xgboost=datos.drop(columns=["score","ranking","themes","title","url","first_emission","last_emission", 'members'])
genres_encoded = datos_xgboost['genres'].str.get_dummies(',')
datos_xgboost.drop(columns="genres",inplace=True)
    
categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, ['demographics', 'emission_type', 'year', 'month','studio']),
        ],
        remainder="passthrough")
    
preprocessor.fit(datos_xgboost)

# save the processor to disk
filename_process = "preprocesamiento_xgboost.pkl"
#Write the scaler to folder
with open(ROOT+"/"+config_f["modelos"]+filename_process, 'wb') as file_process:
    pickle.dump(preprocessor, file_process)

data_processed = preprocessor.transform(datos_xgboost)

data_processed= data_processed.toarray()

X_train, X_test, y_train, y_test = train_test_split(data_processed,y, test_size=0.3, random_state=42)

X_test.tofile(os.path.join(ROOT,config_f["data"]["preprocesados_x"]), sep = ',')
print(type(y_test))
y_test.to_csv(os.path.join(ROOT,config_f["data"]["preprocesados_y"]), sep = ',')


model = xgb.XGBRegressor(objective ='reg:squarederror')

param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'learning_rate': [0.1, 0.01],
    'max_depth': [5, 10],
    'alpha': [1, 10],
    'n_estimators': [10, 100]
}
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

print("Mejores parámetros: ", grid_search.best_params_)
best_grid = grid_search.best_estimator_
predictions = best_grid.predict(X_test)

mse = mean_squared_error(y_test, predictions)

# Best performing model and its corresponding hyperparameters
parameters_best_model=grid_search.best_params_

# save  best model params to disk
filename_bm_params = "parametros_best_xg.pkl"

#Write the best model params to folder
with open(ROOT+ "/"+config_f["modelos"]+filename_bm_params, 'wb') as file_params:
    pickle.dump(parameters_best_model, file_params)

# save  best GS model to disk
filename_bm= "mejor_modelo.pkl"

#Write the best model to folder
with open(ROOT+ "/"+config_f["modelos"]+filename_bm, 'wb') as file_bm:
    pickle.dump(best_grid, file_bm)

