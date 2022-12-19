import streamlit as st

st.title("❗ Introductions")
st.sidebar.markdown("# ❗ Introductions")

intro_objective = """<br/>
<h5> Objective: </h5>
<li> Classify the given genetic variations/mutations based on evidence from text-based clinical literature.</li>
<li> There are nine different classes a genetic mutation can be classified into. Which makes it Multi-class classification problem.</li>
"""

intro_kpis_1 = """
<h5> KPIs: </h5>
<ol start=1><li><a href='https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd'>Multi-Class Log loss metric</a></li>
    <ul>
        <li>It is required by Kaggle, which will be our primary metric.</li>
        <li>It ranges between [0,∞). As usual we want to minimize the loss function given. To compare models, we will compare Loss values. More Lower the more better.</li>
        <li>Penalizes the misclassification.</li>
    </ul>
</ol>
"""

logloss_str = """$L_{logloss}(y,p) = -\sum_{c=1}^{c=M}{y\log(p) + (1 - y)\log(1 - p)}$"""

kpi1_cap = """
<li>Where,
    <ul>
        <ul>
            <ul>
                <li>M - number of classes</li>
                <li>log - the natural log</li>
                <li>y - binary indicator (0 or 1) if class label c is the correct classification for observation o</li>
                <li>p - predicted probability observation o is of class c</li>
            </ul>
        </ul>
    </ul>
</li>
"""

intro_kpis_2 = """
<ol start=2><li> <a href='https://en.wikipedia.org/wiki/Confusion_matrix'> Confusion Matrix </a></li>
    <ul>
        <li> We want to validate it is not making more mistakes. Mistakes are costly here.</li>
        <li> Analyze classwise performance details.</li>
    </ul>
</ol>
"""

intro_constrains = """
<h5>Real-world Constrains:</h5>
    <li>No low-latency requirement.</li>
    <li>Interpretability is important.</li>
    <li>Errors can be very costly. Which is why probability of a data-point belonging to each class is needed. If probability is over threshold then only provide confirmation otherwise needed another expert review for datapoint.</li>
<br/>
"""


# Arrangements
st.write(intro_objective,unsafe_allow_html=True)

st.write(intro_kpis_1,unsafe_allow_html=True)
st.write(logloss_str)
st.caption(kpi1_cap,unsafe_allow_html=True)

st.write(intro_kpis_2,unsafe_allow_html=True)

st.write(intro_constrains,unsafe_allow_html=True)
