#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 12:49:51 2022

@author: florian
"""

import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from statsmodels.formula.api import ols

import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(
    page_title="Econometrics: simple linear regression",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)  


""" ## Simple linear regression"""

c1,c2,c3 = st.columns(3,gap="large")

with c1: 
    st.markdown(""" In this illustration you can run a linear regression model based on the equation: $y = \\alpha +\\beta x + u$. 
                
    There are a few things you can try: 
        
    - Increase and decrease the standard deviation of the erros to see how the confidence intervall of the regression changes
    - Change the number of observations to see how the precision of the estimation increases with the number of observations""")
    
with c2: 
    seed = st.number_input("Random seed (put 0 if you want to have different random number each time)",min_value=0,max_value=100000,value=0)
    n = st.slider("N (number of observations)",min_value=3,max_value=400,value=100)
with c3: 
    alpha = st.slider("$\\alpha$ (constant term)",min_value=-5,max_value=15,value=3,key="alpha")
    beta =st.slider("$\\beta$ (slope)",min_value=-5,max_value=10,value=4)
    sigma = st.number_input("$\\sigma$ (standard deviation of error/residuals",value=0.25,min_value=0.0,max_value=2.0,step=0.1)
    
  

st.markdown("### Results")
 
if seed >0 : 
    np.random.seed(seed)
## continuous explanatory variables
df = pd.DataFrame(np.random.randint(2,100,size=(n, 1)), columns=["x"]) 
#df['e'] = np.random.normal(0,sigma,n)


## GENERATE THE DATA
df['y'] = df.eval(" @alpha + @beta * x")

df['e'] = np.random.normal(0,abs(sigma*df.y.mean()),n)
df['ytrue'] = df.y
df.y = df.y + df.e

## ESTIMATE THE MODEL 
df.sort_values('x',inplace=True)

model = ols("y ~ x",data=df).fit()

pred = model.get_prediction().summary_frame()

## GRAPHICAL OUTPUT
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.x,y=df.y,name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig.add_trace(go.Scatter(x = df.x, y = df.ytrue,name='True model',mode='lines',line=dict(color='red', width=2)))

fig.add_trace(go.Scatter(x = df.x, y = pred['mean'],name='Regression line',mode='lines',line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(
        name='95% CI Upper',
        x=df.x,
        y=pred['mean_ci_upper'],
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ))
fig.add_trace(go.Scatter(
        name='95% CI Lower',
        x=df.x,
        y=pred['mean_ci_lower'],
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(0,0,255 0.3)',
        fill='tonexty',
        showlegend=False
    ))

#fig = px.strip(df,x="mean",y="true",color='conclusion',range_x=(0,1),
#                   labels={"conclusion":"Conclusion of test","true":"Actual coin used","mean":"Proportion of heads (H)"})
 #   fig.add_vline(x=0.5,line_width=1, line_dash="solid", line_color="gray")
#    fig.add_vline(x=mu,line_width=1, line_dash="solid", line_color="gray")
    
#fig = px.scatter(df,y="y",x="x",symbol={"color":'red'})
c1,c2 = st.columns(2)
c1.plotly_chart(fig,use_container_width=True)


    
with c2: 
    fig2 = go.Figure();
    betas = model.params
    betas.name="Betas"
    ci = model.conf_int(alpha=0.05)
    
    res = ci.join(betas).reset_index().rename(columns={0:"CI_lower",1:"CI_upper"})
    
    res['ci_width'] = res['Betas'] - res['CI_lower']
    
    res = res.query("index=='x'")
    
    #st.write(res)
    fig = px.scatter(res,y="index",x="Betas",error_x="ci_width",range_x=(-5,15))
    fig.add_vline(beta,line_width=1, line_dash="solid", line_color="gray",name="True value")
    fig.update_layout(yaxis_range=[-4,4])
    st.plotly_chart(fig,use_container_width=True)
