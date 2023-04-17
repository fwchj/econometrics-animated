#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 05:42:53 2022

@author: florian
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
st.set_page_config(
    page_title="Basic econometrics illustrated",
    page_icon="chart_with_upwards_trend",
    layout="wide",
)  

st.image("resources/coreso.png", width=150)
st.write("# Introductory econometrics - illustrated and animated")
         
"""
#### The goal of this app is to provide students of introdcutory econometrics courses with an intuitive and interactive tool to better understand some core concepts of econometric modelling. 

#### Each illustration is dedicated to a specific topic and users can adjust parameters to see how the model behaves.

##### :point_left: use the menu on the left to select an illustration


This app is being developped by Florian Chávez from CORESO - Collaboartive Research Solutions. For suggestions and feedback, please feel free to reach out by e-mail to econometrics.app@coreso.ch

"""        



#st.sidebar.image("resources/coreso.png", width=100)

