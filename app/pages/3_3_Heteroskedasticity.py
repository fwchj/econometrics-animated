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

""" ## Heteroskedasticity and robust standard errors"""

c1,c2,c3 = st.columns(3,gap="large")

with c1: 
    st.markdown(""" When using real-world data, the hypothesis of homoscedastic error terms is rarely satisfied and therefore the 
                estimated standard errors are biased. On this page you can see how the heteroskedasticity-robust standard errors behave depending 
                on how the errors are distributed.
                
1. Select the type of heteroskedasticity
2. Compare the robust and the non-robust standard error. What do you observe? """)
    
with c2: 
    st.markdown(f"""True model: $y_i = {alpha} + {beta}x_i + u_i$""")
    seed = st.number_input("Random seed (put 0 if you want to have different random number each time)",min_value=0,max_value=100000,value=0)
    
with c3: 
    st.markdown("""Select the type of behaviour for the error:""")
    
    hetoptions = {"homo":"Homoscedastic error terms",
                  "increasing":"Error variance increasing with x",
                  "decreasing":"Error variance decreasing with x",
                  "extremes":"Higher error variance for low and high values of x",
                  "center":"Higher error variance in the middle of x"}
    
    errortype = st.radio('Which behaviour of the error variance you want to simulate?',
                             options=hetoptions.keys(),format_func=lambda x: hetoptions[x] )

    
  

st.markdown("### Results")
 
if seed >0 : 
    np.random.seed(seed)

## continuous explanatory variables
n = 80
df = pd.DataFrame(np.random.randint(2,100,size=(n, 1)), columns=["x"]) 
#df['e'] = np.random.normal(0,sigma,n)


df['y'] = df.eval(" @alpha + @beta * x")
df['ytrue'] = df['y']
if errortype=="increasing":
    df['e'] = np.random.normal(0,abs(df.x),n)
elif errortype=="decreasing":
    df['e'] = np.random.normal(0,abs(df.x.max()-df.x),n)
elif errortype=="extremes":
    df['e'] = np.random.normal(0,abs(df.x.mean()-df.x),n)
elif errortype=="center":
    df['dmax'] = 100 - df.x
    df['dmin'] = df.x - 2
    df['dext'] = np.min(df[['dmin','dmax']],axis=1)
 
    df['e'] = np.random.normal(0,abs(df.dext),n) #np.random.normal(0,abs(np.min(df.x-df.x.min(),df.x.max()-df.x),axis=1),n)
else:
    df['e'] = np.random.normal(0,abs(25),n)
    
df.y = df.y + df.e

##### MODEL ESTIMATION ========================================================================
df.sort_values('x',inplace=True)

model = ols("y ~ x",data=df).fit()
model_robust = ols("y ~ x",data=df).fit(cov_type="HC0")

pred = model.get_prediction().summary_frame()
pred_robust = model_robust.get_prediction().summary_frame()

##### GRAPHICAL OUTPUT ========================================================================
## GRAPHICAL OUTPUT
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.x,y=df.y,name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig.add_trace(go.Scatter(x = df.x, y = df.ytrue,name='True model',mode='lines',line=dict(color='black', width=4)))

#st.dataframe(pred)
fig.add_trace(go.Scatter(
        name='95% CI Upper',
        x=df.x,
        y=pred_robust['mean_ci_upper'],
        mode='lines',
        marker=dict(color='#444'),
        line=dict(width=0),
        showlegend=False
    ))
fig.add_trace(go.Scatter(
        name='95% CI (robust)',
        x=df.x,
        y=pred_robust['mean_ci_lower'],
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(0,0,255,0.5)',
        fill='tonexty',
        showlegend=True
    ))
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
        name='95% CI (non-robust)',
        x=df.x,
        y=pred['mean_ci_lower'],
        marker=dict(color='#444'),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(255,0,0 0.3)',
        fill='tonexty',
        showlegend=True
    ))
        
fig.add_trace(go.Scatter(x = df.x, y = pred_robust['mean'],name='Regression line (robust)',mode='lines',line=dict(color='blue', width=4)))        

        

fig.add_trace(go.Scatter(x = df.x, y = pred['mean'],name='Regression line',mode='lines',line=dict(color='red', dash= 'dot',width=4)))
fig.update_layout(
    title="Robust vs non-robust OLS",
    xaxis_title="x",
    yaxis_title="y",
    
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df.x,y=abs(df.y-pred['mean']),name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig2.update_layout(
    title="Residuals (absolute)",
    xaxis_title="x",
    yaxis_title="Absolute value of estimated residual",
    
)


#fig = px.strip(df,x="mean",y="true",color='conclusion',range_x=(0,1),
#                   labels={"conclusion":"Conclusion of test","true":"Actual coin used","mean":"Proportion of heads (H)"})
 #   fig.add_vline(x=0.5,line_width=1, line_dash="solid", line_color="gray")
#    fig.add_vline(x=mu,line_width=1, line_dash="solid", line_color="gray")
    
#fig = px.scatter(df,y="y",x="x",symbol={"color":'red'})
c1,c2 = st.columns((3,2))
c1.plotly_chart(fig,use_container_width=True)
c2.plotly_chart(fig2,use_container_width=True)

c1,c2 = st.columns(2)

with c1: 
    st.markdown("""### Non robust estimation""")
    s1 = model.summary()
    s1
with c2: 
    st.markdown("""### Robust estimation""")
    s2 = model_robust.summary()
    s2
#fig,ax = plt.subplots(ncols=1,figsize=(8,3))

#df.plot(x="x", y=["y"],kind="scatter",ax=ax)
#st.pyplot(fig)