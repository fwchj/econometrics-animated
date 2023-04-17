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

""" ## Back-transformation of transformed dependent variable"""

if 'data' not in st.session_state:
    st.write("Session state (data) not yet set")
    st.session_state['data'] =[]

c1,c2 = st.columns((4,4))
c1.markdown(""" Whenever the dependent variable of a linear model has been transformed with a non-linear function (e.g. natural logarithm), obtaining the expected value in levels is not trivial, because
simply applying the inverse function would yield wrong values. On this page we illustrate this issue with a log-linear model by showing how much the expected values are biased and how
the correction by Duan can help reduce the bias.
""")

c1.markdown("$E[y] = E[exp(ln(y))] = E\\left[exp(X\\beta + u)\\right] \\ \\neq E\\left[exp(X\\beta)\\right]$")
varoptions = {10:"Very small variance (Very high R-squared)",
              30:"Small variance",
              50:"Medium variance",
              75:"High variance",
             100:"Very high variance",
             200:"Extreme variance (Very low R-squared)"}
    
sigma = c2.radio('Select the variance of the model',
                             options=varoptions.keys(),format_func=lambda x: varoptions[x] )

st.markdown("---")
###### GENERATE THE DATA ############################################################
n = 100
df = pd.DataFrame(np.random.randint(10,160,size=(n, 1)), columns=["x"]) 
df.x = df.x / 10
df.sort_values("x",inplace=True)
df['y'] = df.eval(" 3500 + 100 * x + 8*x*x")
df['e'] = np.random.normal(0,sigma*abs(df.x),n)
df.y = df.y + df.e
df['lnY'] = np.log(df.y)


###### ESTIMATE THE MODEL ############################################################

model_1 = ols("y ~ x + I(x**2)",data=df).fit()
pred_1 = model_1.get_prediction().summary_frame()

model_2 = ols("lnY ~ x ",data=df).fit()
pred_2 = model_2.get_prediction().summary_frame()


###### GRAPH (of the model) ############################################################



fig = go.Figure()
fig.add_trace(go.Scatter(x=df.x,y=df.y,name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
fig.add_trace(go.Scatter(x = df.x, y = pred_1['mean'],name='Regression line (quadratic)',mode='lines',line=dict(color='blue', width=4)))        
fig.add_trace(go.Scatter(x = df.x, y = np.exp(pred_2['mean']),name='Regression line (log-linear)',mode='lines',line=dict(color='red', width=4)))        
#fig.add_shape(type='line',x0=df.x.min(),x1=df.x.max(),y0=pred_1['mean'].mean(),y1=pred_1['mean'].mean(),line={'color':'blue'},name='test',legend=True)
#fig.add_hline(y=pred_1['mean'].mean(), line_width=3, line_dash="dash", name='test',line_color="green")

fig.add_trace(
    go.Scatter(
        mode='lines',
        name='E[y | x] with quadratic model',
        x=df['x'],
        y=[pred_1['mean'].mean()]*len(df),
        line_color='blue',
        line={"dash":"dash"},
        legendgroup='1'
    )
)
    
fig.add_trace(
    go.Scatter(
        mode='lines',
        name='E[y | x] with log-linear model',
        x=df['x'],
        y=[np.exp(pred_2['mean']).mean()]*len(df),
        line_color='red',
        line={"dash":"dot"},
        legendgroup='1'
    )
)
fig.update_layout(
    title="OLS estimation",
    xaxis_title="x",
    yaxis_title="y",
    
)

c1,c2 = st.columns(2)
c1.plotly_chart(fig,use_container_width=True)

####### STATISTICS ON THE RIGHT ###########################################

ey_correct = pred_1['mean'].mean()
ey_log = np.exp(pred_2['mean']).mean()
result = dict()
result['Observations'] = dict()
result['Observations']['E[y]'] = df.y.mean()
result['Observations']['R squared'] = "---"
result['Quadratic model'] = dict()
result['Quadratic model']['E[y]'] = ey_correct
result['Quadratic model']['R squared'] = "%.2f" % model_1.rsquared
result['Log-linear model'] = dict()
result['Log-linear model']['E[y]'] = np.exp(pred_2['mean']).mean()
result['Log-linear model']['R squared'] = "%.2f" % model_2.rsquared 



result = pd.DataFrame().from_dict(result).T.reset_index().rename(columns={"index":"Statistic"})
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)
c2.table(result)

diff_rel = 100* abs(ey_log / ey_correct - 1)
diff_abs = ey_correct - ey_log

c2.write("The predicted average is %.2f%%  (%.1f units) below the actual average" % (diff_rel,diff_abs))

c2.markdown("---")
########## GRAPH WITH ALL THE RESULTS ##############
st.session_state['data'].append([model_2.rsquared,diff_rel])

with c2: 
#    st.session_state
    stat = pd.DataFrame(st.session_state['data'],columns=['R','error'])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stat.R,y=stat.error,mode='markers',marker_color='rgba(0,0,0,0.8)'))
    fig.update_layout(
            title="Underestimation as a function of the model fit (R-squared)",
            xaxis_title="R squared of the log-linear model",
            yaxis_title="Understimation in %",
    
    )
    st.plotly_chart(fig)
#c1,c2,c3 = st.columns(3,gap="large")
#
#with c1: 
#    st.markdown(""" When using real-world data, the hypothesis of homoscedastic error terms is rarely satisfied and therefore the 
#                estimated standard errors are biased. On this page you can see how the heteroskedasticity-robust standard errors behave depending 
#                on how the errors are distributed.
#                
#1. Select the type of heteroskedasticity
#2. Compare the robust and the non-robust standard error. What do you observe? """)
#    
#with c2: 
#    st.markdown(f"""True model: $y_i = {alpha} + {beta}x_i + u_i$""")
#    seed = st.number_input("Random seed (put 0 if you want to have different random number each time)",min_value=0,max_value=100000,value=0)
#    
#with c3: 
#    st.markdown("""Select the type of behaviour for the error:""")
#    
#    hetoptions = {"homo":"Homoscedastic error terms",
#                  "increasing":"Error variance increasing with x",
#                  "decreasing":"Error variance decreasing with x",
#                  "extremes":"Higher error variance for low and high values of x",
#                  "center":"Higher error variance in the middle of x"}
#    
#    errortype = st.radio('Which behaviour of the error variance you want to simulate?',
#                             options=hetoptions.keys(),format_func=lambda x: hetoptions[x] )
#
#    
#  
#
#st.markdown("### Results")
# 
#if seed >0 : 
#    np.random.seed(seed)
#
### continuous explanatory variables
#n = 100
#df = pd.DataFrame(np.random.randint(2,100,size=(n, 1)), columns=["x"]) 
##df['e'] = np.random.normal(0,sigma,n)
#
#
#df['y'] = df.eval(" @alpha + @beta * x")
#df['ytrue'] = df['y']
#if errortype=="increasing":
#    df['e'] = np.random.normal(0,abs(df.x),n)
#elif errortype=="decreasing":
#    df['e'] = np.random.normal(0,abs(df.x.max()-df.x),n)
#elif errortype=="extremes":
#    df['e'] = np.random.normal(0,abs(df.x.mean()-df.x),n)
#elif errortype=="center":
#    df['dmax'] = 100 - df.x
#    df['dmin'] = df.x - 2
#    df['dext'] = np.min(df[['dmin','dmax']],axis=1)
# 
#    df['e'] = np.random.normal(0,abs(df.dext),n) #np.random.normal(0,abs(np.min(df.x-df.x.min(),df.x.max()-df.x),axis=1),n)
#else:
#    df['e'] = np.random.normal(0,abs(25),n)
#    
#df.y = df.y + df.e
#
###### MODEL ESTIMATION ========================================================================
#df.sort_values('x',inplace=True)
#
#model = ols("y ~ x",data=df).fit()
#model_robust = ols("y ~ x",data=df).fit(cov_type="HC0")
#
#pred = model.get_prediction().summary_frame()
#pred_robust = model_robust.get_prediction().summary_frame()
#
###### GRAPHICAL OUTPUT ========================================================================
### GRAPHICAL OUTPUT
#fig = go.Figure()
#fig.add_trace(go.Scatter(x=df.x,y=df.y,name='Observations',mode='markers',marker_color='rgba(0,0,0,0.3)'))
#fig.add_trace(go.Scatter(x = df.x, y = df.ytrue,name='True model',mode='lines',line=dict(color='black', width=4)))
#
##st.dataframe(pred)
#fig.add_trace(go.Scatter(
#        name='95% CI Upper',
#        x=df.x,
#        y=pred_robust['mean_ci_upper'],
#        mode='lines',
#        marker=dict(color='#444'),
#        line=dict(width=0),
#        showlegend=False
#    ))
#fig.add_trace(go.Scatter(
#        name='95% CI (robust)',
#        x=df.x,
#        y=pred_robust['mean_ci_lower'],
#        marker=dict(color='#444'),
#        line=dict(width=0),
#        mode='lines',
#        fillcolor='rgba(0,0,255,0.5)',
#        fill='tonexty',
#        showlegend=True
#    ))
#fig.add_trace(go.Scatter(
#        name='95% CI Upper',
#        x=df.x,
#        y=pred['mean_ci_upper'],
#        mode='lines',
#        marker=dict(color='#444'),
#        line=dict(width=0),
#        showlegend=False
#    ))
#fig.add_trace(go.Scatter(
#        name='95% CI (non-robust)',
#        x=df.x,
#        y=pred['mean_ci_lower'],
#        marker=dict(color='#444'),
#        line=dict(width=0),
#        mode='lines',
#        fillcolor='rgba(255,0,0 0.3)',
#        fill='tonexty',
#        showlegend=True
#    ))
#        
#fig.add_trace(go.Scatter(x = df.x, y = pred_robust['mean'],name='Regression line (robust)',mode='lines',line=dict(color='blue', width=4)))        
#
#        
#
#fig.add_trace(go.Scatter(x = df.x, y = pred['mean'],name='Regression line',mode='lines',line=dict(color='red', dash= 'dot',width=4)))
#fig.update_layout(
#    title="Robust vs non-robust OLS",
#    xaxis_title="x",
#    yaxis_title="y",
#    
#)
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
#c1,c2 = st.columns((3,2))
#c1.plotly_chart(fig,use_container_width=True)
#c2.plotly_chart(fig2,use_container_width=True)
#
#c1,c2 = st.columns(2)
#
#with c1: 
#    st.markdown("""### Non robust estimation""")
#    s1 = model.summary()
#    s1
#with c2: 
#    st.markdown("""### Robust estimation""")
#    s2 = model_robust.summary()
#    s2
##fig,ax = plt.subplots(ncols=1,figsize=(8,3))
#
##df.plot(x="x", y=["y"],kind="scatter",ax=ax)
##st.pyplot(fig)