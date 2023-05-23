# Anime Website

Este repositorio contiene la segunda parte del proyecto de la materia de arquitectura de datos de la MCD. 

Aqui implementamos la parte de analisis y desplegamos una pagina web en un entorno de Quarto. 

Para consultar la primera parte del proyecto en la cual hacemos webscrapping de la pagina principal de anime y armamos el ETL en AWS favor de consultar el siguiente repositorio: 

https://github.com/JuanPalms/Anime_ELT_dashboard


``` bash


Anime Website
├── AnimeAnalysis.yaml  # archivo de entorno conda para el proyecto
├── config.yaml # 
├── data
├── introduction.md
├── modelos
│   ├── datos_preprocesados.pkl
│   ├── mejor_modelo.pkl
│   ├── optimal_clusters.png
│   ├── parametros_best_xg.pkl
│   └── preprocesamiento_xgboost.pkl
├── _quarto.yml
├── README.md
├── _site
│   ├── index.html
│   ├── introduction.html
│   ├── src
│   └── styles.css
├── src
│   ├── clustering.py
│   ├── exploracion_inicial.ipynb
│   ├── images
│   ├── preprocessing_cluster.py
│   ├── preprocessing_data_training.py
│   ├── top_lists.ipynb
│   ├── utils.py
│   └── xgboost_predictions.ipynb
└── styles.css

```


