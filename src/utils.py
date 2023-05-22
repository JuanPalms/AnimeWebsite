import os
import yaml
import logging
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from re import sub
from ast import literal_eval

CURRENT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CURRENT)

# Function to load yaml configuration file
def load_config(config_name):
    """
    Sets the configuration file path
    Args:
    config_name: Name of the configuration file in the directory
    Returns:
    Configuration file
    """
    with open(os.path.join(ROOT, config_name), encoding="utf-8") as conf:
        config = yaml.safe_load(conf)
    return config


def global_format(x):
    '''
    Remove atypical character from given string

    Params:
        x: String to format
    
    Returns:
        x: String without atypical characters
    '''
    x = sub(pattern=r"[()°.-]+", repl='_', string=x)
    x = sub(pattern="'s", repl='', string=x)
    return x

# lambda to format single string in dataframe cell
format_single_str = lambda cel_: '_'.join(
    cel_.rstrip().lstrip().split(sep=' ')
).lower()

# lambda to format list like strings in dataframe cell
format_list_str = lambda cel_: [
    '_'.join(
        global_format(val_.rstrip().lstrip())
        .lower()
        .split(' ')
    ) for val_ in cel_.split(',')
]


def tabulacion(df, selected_columns, top_margin=10, bottom_margin=10, left_margin=50, right_margin=50, w=500, h=500):
    fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>{}</b>'.format(i) for i in selected_columns],
        line_color='darkslategray',
        fill_color='royalblue',
        align='left',
        font=dict(color='white', size=12, family="Arial")
    ),
    cells=dict(
        values=[df[col] for col in selected_columns],
        line_color='darkslategray',
        fill=dict(color=['lightgrey', 'white']),
        align='left',
        font=dict(color='darkslategray', size=11, family="Arial"),
        height=30
    ))
    ])

    fig.update_layout(
        autosize=False,
        width=w,
        height=h,
        margin=dict(
            l=left_margin,
            r=right_margin,
            b=bottom_margin,
            t=top_margin,
            pad=5
        ),
        paper_bgcolor='lightgrey'
    )
    
    fig.show()

def apply_format(df):
    df2=df\
    .assign(
        members = lambda df_: df_.members.map('{:,.0f}'.format),
        number_episodes = lambda df_: df_['number_episodes'].apply(lambda x: pd.to_numeric(x, errors='coerce', downcast='integer')).map('{:.0f}'.format),
        year = lambda df_: df_['year'].astype(int, errors='ignore').map('{:.0f}'.format)
        )\
    .rename(columns={"score": "Score promedio ","title":"Título","studio":"Estudio",
                     "themes":"Temática","demographics":"Público objetivo", "number_episodes":"Episodios", 
                     "members": "Popularidad", "genres": "Género","year":"Año"})
    return(df2)



def impute_missing_drop_columns(df,numerical,nominal):
    """
    Function to impute missing values with median or mode to features containing less
    than 50% of missing values and drops the columns with more than 50% of missing values
    Args:
        df (pandas dataframe): dataframe
        numerical (list):   list of numerical attributes
        nominal   (list):   list of nominal attributes
    Returns:
        df (pandas dataframe): dataframe
        numerical (list):   modified list of numerical attributes
        nominal   (list):   modified list of nominal attributes
        """
    # check for null values
    null_val = (df.isnull()
                # sum of null values
                    .sum()
                # sort values in descending order
                    .sort_values(ascending=False)
                # reset index
                    .reset_index())
    #rename columns
    null_val.columns = ['attribute','count']
    #Add percentage of null values column
    null_val['percentage']=(null_val['count']/df.shape[0])


    # iterate through df to impute median or mode
    to_remove=[]
    for index,row in null_val.iterrows():
        # if percentage of nulls between 0 and 30%
        if 0<row["percentage"]<=.3:
            # print attribute and percentage of nulls
            logging.debug(f'{row["attribute"]} has {row["percentage"]*100} percent of null values')
            #deal with numerical attributes
            if row["attribute"] in numerical:
                # impute median
                df.loc[(df[row['attribute']].isnull()==True),
                       row['attribute']]=df[row['attribute']].median()
            # impute mode to nominal and ordinal variables
            elif row["attribute"] not in numerical:
                df.loc[(df[row['attribute']].isnull()==True),
                       row['attribute']]=df[row["attribute"]].mode(dropna=True)[0]
                # if percentage is greater than 30% remove columns
        elif row["percentage"]>.3:
            to_remove.append(row["attribute"])
        else:
            continue

    remove = set(to_remove)
    numerical = [x for x in numerical if x not in remove]
    nominal = [x for x in nominal if x not in remove]
    df.drop(columns=to_remove, inplace=True)


    return df,numerical,nominal
