import pandas as pd, numpy as np
from scipy.stats import uniform, bernoulli, beta
from random import choices, sample
from itertools import combinations
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
#import pygraphviz as pgv
import streamlit as st
import streamlit.components.v1 as components

PAGE_CONFIG = {"page_title":"Covid-19 simulation","page_icon":":mask:","layout":"wide"}
st.set_page_config(**PAGE_CONFIG)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def main():
    
    st.subheader("TEST1")
    #st.write(plt.plot(df[1:]))
    #st.write(com.render_community_graph())
    
if __name__ == '__main__':
        main()
