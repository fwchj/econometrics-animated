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

alpha = 3
beta = 2

""" ## Measurement error """

""" When using real-world data, we often face the problem of **measurement errors**. A measurement errors occures when the observed variable is composed of
the true value and an error. For simplicity, let us assume that the true model is 
$y = \\alpha + \\beta w + u$.  Unfortnately, we cannot observe $w$, but only $x = w + v$, where $v$ is the measurement error. We assume $v$ to be independent of 
$w$ and to have $E[v] = 0$

**What happens if we try to estimate $\\beta$ with $x$?**  $\\ \\ \\rightarrow \\hat{\\beta}$ will be biased towards zero (i.e. $|\\hat{\\beta}| < |\\beta|)$

Let us illustrate this with a simple example: """





c1,c2,c3,c4,c5 = st.columns(5,gap="large")

alpha   = c1.number_input("$\\alpha$",min_value=-2,max_value=2,value=1)
beta    = c2.number_input("$\\beta$",min_value=-5,max_value=5,value=2)
sigma_u = c3.number_input("$\\sigma_u$",min_value=0,max_value=5,value=1)
sigma_v = c4.number_input("$\\sigma_v$",min_value=0,max_value=5,value=1)
seed    = c5.number_input("Random seed",min_value=0,max_value=100000,value=0)

st.markdown("### Results")
 
if seed >0 : 
    np.random.seed(seed)

## continuous explanatory variables
n = 100
df = pd.DataFrame(np.random.randint(2,100,size=(n, 1)), columns=["w"]) 
df.w = df.w * 0.1
df['u'] = np.random.normal(0,sigma_u,n)
df['v'] = np.random.normal(0,sigma_v,n)

df['x'] = df['w'] + df['v']
df.eval("y = @alpha + @beta * w + u",inplace=True)




##### MODEL ESTIMATION ========================================================================
df.sort_values('x',inplace=True)

model_true = ols("y ~ w",data=df).fit()
pred_true = model_true.get_prediction().summary_frame()

model_est = ols("y ~ x ",data=df).fit()
pred_est = model_est.get_prediction().summary_frame()

##### GRAPHICAL OUTPUT ========================================================================
## GRAPHICAL OUTPUT
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.w,y=df.y,name='w',mode='markers',marker_color='rgba(255,0,0,1)'))
fig.add_trace(go.Scatter(x=df.x,y=df.y,name='x',mode='markers',marker_color='rgba(0,0,255,1)'))

fig.add_trace(go.Scatter(x = df.w, y = pred_true['mean'],name='Regression line (w)',mode='lines',line=dict(color='red',width=4)))

fig.add_trace(go.Scatter(x = df.x, y = pred_est['mean'],name='Regression line (x)',mode='lines',line=dict(color='blue',width=4)))

fig.update_layout(
    title="y against w and x",
    xaxis_title="w",
    yaxis_title="y",
    showlegend=True
    )

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.v,y=df.y,name='w',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig2.update_layout(
    title="y against v",
    xaxis_title="v",
    yaxis_title="y",
    showlegend=False
    )

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df.x,y=df.y,name='x',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig3.add_trace(go.Scatter(x = df.x, y = pred_est['mean'],name='Regression line',mode='lines',line=dict(color='red',width=4)))
fig3.update_layout(
    title="y against x",
    xaxis_title="x",
    yaxis_title="y",
    showlegend=False
    )
#
#fig2 = go.Figure()
#fig2.add_trace(go.Scatter(x=df.x,y=abs(df.y-pred['mean']),name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
#fig2.update_layout(
#    title="Residuals (absolute)",
#    xaxis_title="x",
#    yaxis_title="Absolute value of estimated residual",
#    
#)
#
#
##fig = px.strip(df,x="mean",y="true",color='conclusion',range_x=(0,1),
##                   labels={"conclusion":"Conclusion of test","true":"Actual coin used","mean":"Proportion of heads (H)"})
# #   fig.add_vline(x=0.5,line_width=1, line_dash="solid", line_color="gray")
##    fig.add_vline(x=mu,line_width=1, line_dash="solid", line_color="gray")
#    
##fig = px.scatter(df,y="y",x="x",symbol={"color":'red'})
c1,c2, = st.columns((5,2))
c1.plotly_chart(fig,use_container_width=True)
c2.plotly_chart(fig2,use_container_width=True)

c1,c2 = st.columns(2)

with c1: 
    st.markdown("""### Estimation with $w$""")
    s1 = model_true.summary()
    s1
with c2: 
    st.markdown("""### Estimation with $x = w + v$""")
    s2 = model_est.summary()
    s2
#fig,ax = plt.subplots(ncols=1,figsize=(8,3))

#df.plot(x="x", y=["y"],kind="scatter",ax=ax)
#st.pyplot(fig)