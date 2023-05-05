from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
import numpy as np
import prophet

 
df = pd.read_csv('verdun_MAJ.csv' )
 
df['y'] = pd.array(df.y, dtype=pd.Int64Dtype())
df['ds'] = pd.to_datetime(df.ds, format='%Y-%m-%d', errors='coerce')

df1 = df.loc[df.ds <"2020-01-01"].copy()
df2 =df.loc[df.ds> "2020-12-31"].copy()
frames = [df1, df2]
split_date = '2023-04-15'  
df = pd.concat(frames)
df_train = df.loc[df.ds <=split_date].copy()
df_test =df.loc[df.ds> split_date].copy()

from prophet import Prophet


def predict(date_future):
 df_train_prophet = df_train.reset_index()  

 

 model = Prophet ( )
 model.add_country_holidays(country_name='FR')

 model.fit(df_train_prophet)
 future_date = pd.date_range(date_future , periods=30, freq='D')
 future_date = pd.DataFrame({'ds': future_date })
 pred = model.predict(future_date )
 return pred

 

from pandas.api.types import CategoricalDtype


def create_features(df, label=None):
     
    df = df.copy()
    df['datetime'] = df['ds']
    
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['ds'] = df.index
    
    X = df[['datetime','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features(df, label='y')


features_and_target = pd.concat([X, y], axis=1)


cat_type = CategoricalDtype(categories=['Monday','Tuesday',
                                        'Wednesday',
                                        'Thursday','Friday',
                                        'Saturday','Sunday'],
                            ordered=True)

def create_features_saison(df, label=None):
  
    
    df = df.copy()
    
     
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['weekday'] = df['ds'].dt.day_name()
    df['weekday'] = df['weekday'].astype(cat_type)
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week
    df['date_offset'] = (df.ds.dt.month*100 + df.ds.dt.day - 320)%1300
    df['date'] = df.index
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], 
                          labels=['Printemps' ,'Été', 'Automne' ,'Hiver']
                   )
    X = df[[ 'dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear','weekday',
           'season']]
    if label:
        y = df[label]
        return X, y
    return X

X, y = create_features_saison(df,label='y')
features_and_target = pd.concat([X, y], axis=1)

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def plot1(date_future ):
 pred= predict(date_future)
 fig = make_subplots(specs=[[{"secondary_y": True}]])
 fig.add_trace(
    go.Scatter(x=pred['ds'], y=pred['yhat'], name="valeurs réelles"),
    secondary_y=False,)
 
 fig.update_layout(
    title_text="valeurs réelles vs Prédites")
 fig.update_xaxes(title_text="Timeline")
 fig.update_yaxes(title_text="valeurs réelles", secondary_y=False)
 return fig
def pilot() :
  sns.pairplot(features_and_target.dropna(),
             hue='year',  palette='hls',
             x_vars=['dayofweek',
                     'year','weekofyear'],
             y_vars='y',
             height=5,
             plot_kws={'alpha':0.15, 'linewidth':0}  
            )
  plt.suptitle('Nombre entrée parking par jour , année et semaine ')
 
  return plt
def pilot3(): 
  fig, ax = plt.subplots(figsize=(10, 5))
  sns.boxplot(data=features_and_target.dropna(),
            x='weekday',
            y='y',
            hue='season',
            ax=ax,
            linewidth=1)
  ax.set_title('Nombre Entrée Parking par saison  ')
  ax.set_xlabel('jour de la semaine')
  ax.set_ylabel('Nombre Entrée Parking')
  ax.legend(bbox_to_anchor=(1, 1))
  return plt 
 
 
 
 
 
 
 
 
 
 
 
st.title("Modèle de prédiction des entrées du parking Verdun Sud")
dfe=st.date_input("date future")

fig1 = pilot() 

st.pyplot(fig1)
fig = pilot3() 

st.pyplot(fig)

dat = st.text_input("Faire la prédiction à partir de la date : ", '') 

if dfe:
 d=predict(dfe)
 st.dataframe(d['yhat'])
 chart_data = pd.DataFrame(d[[ 'ds','yhat' ]] )
 st.line_chart(chart_data , x='ds', y='yhat' )




