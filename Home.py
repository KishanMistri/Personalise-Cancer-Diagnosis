import streamlit as st
import pandas as pd
import numpy as np
import os

st.title('🩺 Personalized Cancer Diagnosis ')

home_content = """<p>
    A lot has been said during the past several years about how precision medicine and, more concretely, how genetic testing is going to disrupt the way diseases like cancer are treated.
</p> 
<p> 
    But this is only partially happening due to the huge amount of manual work still required. Memorial Sloan Kettering Cancer Center (MSKCC) launched this competition, accepted by the NIPS 2017 Competition Track,  because we need your help to take personalized medicine to its full potential. Once sequenced, a cancer tumor can have thousands of genetic mutations. But the challenge is distinguishing the mutations that contribute to tumor growth (drivers) from the neutral mutations (passengers). Currently this interpretation of genetic mutations is being done manually. This is a very time-consuming task where a clinical pathologist has to manually review and classify every single genetic mutation based on evidence from text-based clinical literature.
</p> 
<p>
    For this competition MSKCC is making available an expert-annotated knowledge base where world-class researchers and oncologists have manually annotated thousands of mutations. We need to develop a Machine Learning algorithm that, using this knowledge base as a baseline, automatically classifies genetic variations.
</p>
For more information:
<li><a href="https://www.kaggle.com/competitions/msk-redefining-cancer-treatment/overview">Problem Overview</a></li>
<li><a href="https://www.kaggle.com/c/msk-redefining-cancer-treatment/discussion/35336#198462">Understanding Workflow</a></li>
<li><a href="https://www.youtube.com/watch?v=qxXRKVompI8">Understand Mutations</a></li>"""

st.write(home_content,unsafe_allow_html=True)

# Just for streamlit cloud deployment. Otherwise the entrypoint will be setup.sh only
os.system("chmod +x ./setup.sh")
os.system("bash ./setup.sh")
