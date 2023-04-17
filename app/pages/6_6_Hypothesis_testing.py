#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:03:17 2023

@author: florian
"""


import streamlit as st 
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Econometrics: simple linear regression",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)  


""" ## Hypothesis testing: illustration with a coin (:coin:)"""
c1,c2,c3 = st.columns(3,gap="large")

with c1: 
    st.markdown(""" In this illustration we show how hypothesis testing works and how likely we are committing errors when concluding based on statitical tests. 
                For simplicity we use a coin and throw it many times. Through a statitical test, we try to figure out if it is a **fair or an unfair coin**""")
                
    st.markdown("Fair coin: $P(H) = P(T) = 0.5$")
    st.markdown("Unfair coin: $P(H) = \mu, P(T) = 1-\mu$")
    st.markdown("$H_0: \mu=0.5,\ \ \ \ H_a: \mu\\neq 0.5$")
                   
    
with c2: 
    seed = st.number_input("Random seed (put 0 if you want to have different random number each time)",min_value=0,max_value=100000,value=0)
    n = st.slider("N (sample size):", help="How many times we throw the coin in each sample. The more we throw it, the smaller will be the variance around the true $\mu$",min_value=5,max_value=1000,value=1000)
    
    alphas = {0.1 : "90% confidence interval",0.05 : "95% confidence interval", 0.01: "99% confidence interval",0.001:"99.9% confidence interval"}
    alpha = st.selectbox("Confidence level",options=alphas.keys(),key='alpha',format_func=lambda x: alphas[x])
with c3: 
    mu = st.slider("$\\mu$ (probabilty of heads when using the unfair coin)",value=0.6,key="mu",min_value=0.00,max_value=1.0,step=0.05)
    n_h0  = st.slider("Number of samples with **fair coin**$ ",min_value=5,max_value=1000,value=50,key="n_h0")
    n_ha  = st.slider("Number of samples with **unfair coin**$ ",min_value=5,max_value=1000,value=50,key="n_ha")



df = pd.DataFrame(columns=["mean","t","pval","true"])

# Samples under H0
for i in range(n_h0 + n_ha):
    thisMu = 0.5 if i<n_h0 else mu
    # Throw the coin n times
    sample = np.random.choice([1,0],n,p=[thisMu,1-thisMu])
    # Run the t-test
    test = stats.ttest_1samp(sample,0.5)
    # Define the true value as text
    muText = "fair" if i<n_h0 else "unfair"

    # Store all the data in the dataframe (for posterior analysis)
    res = pd.DataFrame([[sample.mean(),test[0],test[1],muText]],columns=["mean","t","pval","true"])
    df = pd.concat([df,res])
    
# Determine the test result
df['test'] = df.pval.apply(lambda x: "test: unfair" if x<alpha else "test: fair")  

result = df.groupby(["true",'test'])['pval'].count()
   

df['conclusion'] = df.pval.apply(lambda x: "unfair coin" if x<alpha else "fair coin")

table = df.groupby(['true','test'])['mean'].count().reset_index()
table['mean'] = table['mean'].astype("int")
table = table.pivot(index='true',columns='test',values="mean").fillna(0)
#table

truePositive = int(table.loc['unfair','test: unfair'])
falsePositive = int(table.loc['fair','test: unfair'])
trueNegative = int(table.loc['fair','test: fair'])
falseNegative = int(table.loc['unfair','test: fair'])

n_all = n_h0+ n_ha


st.markdown("## Key results:")
c = st.columns(7)
c[0].metric(label="True positive",value=truePositive,delta="%.2f%%" %(100*truePositive/n_all), delta_color='off')
c[1].metric(label="True negative",value=trueNegative,delta="%.2f%%" %(100*trueNegative/n_all), delta_color='off')
c[2].metric(label="False positive",value=falsePositive,delta="%.2f%%" %(100*falsePositive/n_all), delta_color='off')
c[3].metric(label="False negative",value=falseNegative,delta="%.2f%%" %(100*falseNegative/n_all), delta_color='off')
c[4].metric(label="Type I  error",value = "%.2f%%" % (falsePositive / (falsePositive + trueNegative)))
c[5].metric(label="Type II  error",value = "%.2f%%" % (falseNegative / (falseNegative + truePositive)))
c[6].metric(label="P(unfair | test => unfair)",value = "%.2f%%" % (truePositive / (truePositive + falsePositive)))




C1,C2,C3 = st.columns(3,gap="small")

with C1: 

    fig = px.strip(df,x="mean",y="true",color='conclusion',range_x=(0,1),
                   labels={"conclusion":"Conclusion of test","true":"Actual coin used","mean":"Proportion of heads (H)"})
    fig.add_vline(x=0.5,line_width=1, line_dash="solid", line_color="gray")
    fig.add_vline(x=mu,line_width=1, line_dash="solid", line_color="gray")

    st.plotly_chart(fig,use_container_width=True)
    
    
with C2: 
    fig2 = px.histogram(df,x="mean",color="true")

    
    x0 = df.query("true=='unfair'")['mean']
    # Add 1 to shift the mean of the Gaussian distribution
    x1 = df.query("true=='fair'")['mean']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0,marker={"color":'red'},name="Unfair coin"))
    fig.add_trace(go.Histogram(x=x1,marker={"color":"green"},name="Fair coin"))
    
    
    
    # Overlay both histograms
    fig.update_layout(barmode='overlay',title="Distribution of the average number of heads (H) per sample")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig,use_container_width=True)
with C3:    
    fig2 = px.histogram(df,x="mean",color="true")

    
    x0 = df.query("true=='unfair'")['pval']
    # Add 1 to shift the mean of the Gaussian distribution
    x1 = df.query("true=='fair'")['pval']
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0,histnorm='probability',marker={"color":'red'},name="Unfair coin"))
    fig.add_trace(go.Histogram(x=x1,histnorm='probability',marker={"color":"green"},name="Fair coin"))
    
    
    
    # Overlay both histograms
    fig.update_layout(barmode='overlay',title="Distribution of the p-value of the t-test")
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    st.plotly_chart(fig,use_container_width=True)