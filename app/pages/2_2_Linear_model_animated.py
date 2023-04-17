#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 07:25:15 2023

@author: florian
"""

import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols

import time
st.set_page_config(
    page_title="Econometrics: animated simple linear regression",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)  

trueBeta = 0.25;

@st.cache_data
def generateDate(seed,trueBeta):
    n = 1000
    np.random.seed(seed)
    df = pd.DataFrame(np.random.randint(2,100,size=(n, 1)), columns=["x"]) 
    df['e'] = np.random.normal(0,15,n)


    df['y'] = df.eval(f" 3 + {trueBeta} * x + e")

    return df

# Define initial settings (only for the first initialisation)    
if "n_last" not in st.session_state: 
    #st.session_state['n'] = 3
    st.session_state['n_last'] = 3 #st.session_state['n']
    st.session_state['pvalues'] = dict()
    st.session_state['est'] = dict()
max_n = 100  



    


#st.markdown("""
#    Ideas: 
#        
#    - Animate the estimation by adding more observations
#    - Use a slider to manually play with the number of observations
#    - Display the scatter plot with the observations and the regression line
#    - chart with the confidence intervall (probably below a normal distribution)
#    - indicate significance at
#    - allow for different levels of alpha
#    """)

  


""" ## Simple linear regression: animated 

This illustration shows you how the precision of an estimation increases with the number of observations and how a few observations can strongly incluence the estimated coefficient when the sample size is small.
"""

c1,c2,c3, c4 = st.columns([1,6,3,1])

with c1: 
    seed = st.number_input("Random seed",value=1234,key='seed')
with c2: 
    n = st.slider("Number of observations",key="n",min_value=3,max_value=max_n,value=3)
with c3: 
    popt = {"animation":"Animate from current state to final state","fly":"Direct display of result"}    
    st.selectbox('Select the display mode',options = popt.keys(),\
                                 format_func= lambda x:popt[x],key='mode')
with c4: 
    speedoptions = {"fast":"Fast","medium":"Medium","slow":"Slow"}    
    st.selectbox('Animation speed',options = speedoptions.keys(),\
                                 format_func= lambda x:speedoptions[x],key='speed')

    

df = generateDate(seed,trueBeta)



# creating a single-element container
placeholder = st.empty()



#if st.session_state['mode']=="fly" : 
#    n_from = st.session_state['n']
#    n_to = st.session_state['n']
#else: 
#    n_from = st.session_state['n_last']
#    n_to = st.session_state['n']


n_start = st.session_state['n_last']
n_end = st.session_state['n']
delta = -1 if n_start>n_end else 1
if st.session_state['mode']=="fly" : 
    n_start = n_end

st.session_state['n_last'] = st.session_state['n']
#st.write(f"from {n_start} to {n_end}")



#with placeholder.container():
#    for i in [1,2,3,4,2,3]: 
#        st.write(i)
#        time.sleep(0.5)
#        placeholder.empty()
   
def clearAll():
    st.session_state['pvalues']=dict()
    st.session_state['n']=3
    st.session_state['n_last']=3
    st.session_state['est']=dict()
    
def genPlot(data,thisN):
    
    
    df = data[:thisN]
    ## Run the regression
    df.sort_values('x',inplace=True)
    model = ols("y ~ x",data=df).fit()
    pred = model.get_prediction(df).summary_frame()
    
    ## Store pvalue
    myDict = st.session_state['pvalues']
    seed = st.session_state['seed']
    myDict[f"{seed}:{thisN}"] = model.pvalues[1]
    st.session_state['pvalues'] = myDict.copy()
    
    ## Store coefficient and confidence interval
    cint = model.conf_int()[1][1] - model.params[1]
    st.session_state['est'][f"{seed}:{thisN}"] = f"{model.params[1]}:{cint}"

    
    fig,ax = plt.subplots(ncols=2,figsize=(12,4))

    
    # LEFT GRAPH
    ax[0].scatter(df.x,df.y,color='gray',marker='x',label="Observations")
    ax[0].plot(df.x,pred['mean'],'r-',label="Regression")
    #ax.plot(df.x,pred['mean_ci_upper'],'r--',label="CI")
    
    ax[0].fill_between(df.x, pred['mean_ci_lower'], pred['mean_ci_upper'],color='red',
                    alpha=0.4,label="95%-KI")
    
    ax[0].legend(loc="upper left")
#    ax[0].set_title(f"{n} observations")
    ax[0].set_xlim((data.x.min(),data.x.max()))
    ax[0].set_ylim((data.y.min(),data.y.max()))
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].grid(color="lightgray") # Adding grid lines with option (not saying it's nice)
   
    
    # RIGHT GRAPH
    est = st.session_state['est']
    est = pd.DataFrame({"key":est.keys(),"coefs":est.values()})
    est[['seed', 'n']] = est['key'].str.split(':', 1, expand=True)
    est[['beta','cint']] = est['coefs'].str.split(":",1,expand=True)
    currentSeed = st.session_state['seed']
    
    est.n = est.n.astype(int)
    est.beta = est.beta.astype(float)
    est.cint = est.cint.astype(float)
    est.seed = est.seed.astype(int)
    est = est.query(f"seed == {currentSeed}")  #Filter to show only the results of the current seed
    est.sort_values("n",inplace=True)
    
    ax[1].errorbar(x=est.n,y=est.beta,yerr=est.cint,label='Confidence interval 95%',color='gray',linestyle='')
    ax[1].hlines(y=trueBeta,xmin=est.n.min(),xmax=est.n.max(),color='red',label=f"True coefficient ({trueBeta})",zorder=2.51)
    ax[1].set_xlabel("Number of observations")
    est['color'] = "#008000"
    est.loc[est.n==thisN,'color'] = "#00ff00"
    ax[1].scatter(x=est.n,y=est.beta,label="Point estimate",zorder=2.5,color=est.color)
    ax[1].legend()
    ax[1].grid(color="lightgray")
#    ax[1].vlines(trueBeta,ymin=-0.5,ymax=1.5,color='gray')
    ax[1].set_xlim((3,100))
    ax[1].set_ylim((trueBeta-0.5,trueBeta+0.5))
#    
    plt.suptitle(f"{n} observations: estimated model: y = %.3f + %.3f x" % (model.params[0],model.params[1]))
    #plt.ylim((2500,6500))
    #df[:n].plot(x="x", y=["y"],kind="scatter",ax=ax)
    return fig


# Define animation speed
intervalTimes = {'slow':1.0,'medium':0.4,'fast':0.001}
interval = intervalTimes[st.session_state['speed']]

  
with st.empty():
    
        
    mylist = np.arange(n_start,n_end+delta,delta).tolist()
    #st.write(mylist)
    for n in mylist: 
        
        st.pyplot(genPlot(df,n))
        #st.write(f"⏳ {seconds} seconds have passed")
        #st.write(f"blabla {seconds}")
        time.sleep(interval)
#        dfs = df[:n]
     

    st.pyplot(genPlot(df,n))

c2,c3 = st.columns([4,4])
with c2: 
    st.button("Clear all (start over)",on_click=clearAll)
    
with c3: 
    #st.write(pd.DataFrame.from_dict(st.session_state['pvalues']),index="A")
    
    pvals = st.session_state['pvalues']
    data = pd.DataFrame({"key":pvals.keys(),"pvalue":pvals.values()})
    
    data[['seed', 'n']] = data['key'].str.split(':', 1, expand=True)
    
    data.n = data.n.astype("int")
    
    data.sort_values("pvalue",inplace=True,ascending=False)
    fig,ax = plt.subplots(ncols=1,figsize=(6,3))
    data['color'] = "Not significant"
    data.loc[data['pvalue']<0.1,'color'] = "Significant at 10%"
    data.loc[data['pvalue']<0.05,'color'] = "Significant at 5%"
    data.loc[data['pvalue']<0.01,'color'] = "Significant at 1%"
    
    hue_dict = {"Not significant":"red","Significant at 10%":"orange","Significant at 5%":"yellow","Significant at 1%":"green"}
    sns.scatterplot(data=data,x="n",y="pvalue",ax=ax,hue="color",palette=hue_dict)
    
    plt.legend(title='Signifiance level', loc='upper right')
    plt.grid()
    plt.xlabel("Number of observations")
    plt.ylabel("P-value of slope coefficient")
    st.pyplot(fig)
        
 