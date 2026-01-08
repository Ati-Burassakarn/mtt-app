{\rtf1\ansi\ansicpg874\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;\f2\fnil\fcharset222 Thonburi;
}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import streamlit as st\
import pandas as pd\
import numpy as np\
import matplotlib.pyplot as plt\
import seaborn as sns\
from scipy.optimize import curve_fit\
from io import BytesIO\
\
# --- Page Config ---\
st.set_page_config(page_title="MTT Assay Analyzer", page_icon="
\f1 \uc0\u55358 \u56810 
\f0 ", layout="wide")\
\
# --- Core Functions ---\
def four_PL(x, A, B, C, D):\
    return D + (A - D) / (1.0 + (x / C)**B)\
\
def create_template():\
    df = pd.DataFrame(\{\
        'Concentration': [0, 1, 10, 100],\
        'Rep1': [1.0, 0.9, 0.5, 0.1],\
        'Rep2': [1.02, 0.88, 0.48, 0.12],\
        'Rep3': [0.98, 0.92, 0.52, 0.09]\
    \})\
    return df\
\
@st.cache_data\
def convert_df(df):\
    return df.to_csv(index=False).encode('utf-8')\
\
def analyze_data(df, blank_od):\
    # Logic 
\f2 \'e0\'b4\'d4\'c1\'a8\'d2\'a1\'a1\'d2\'c3\'a4\'d3\'b9\'c7\'b3
\f0 \
    control_row = df[df['Concentration'] == 0]\
    if control_row.empty:\
        raw_control = df.iloc[:, 1:].max().mean()\
    else:\
        raw_control = control_row.iloc[:, 1:].values.mean()\
    \
    corrected_control = raw_control - blank_od\
    \
    stats = []\
    for _, row in df.iterrows():\
        conc = row['Concentration']\
        reps = row.iloc[1:].values.astype(float)\
        mean_od = np.nanmean(reps)\
        sd_od = np.nanstd(reps, ddof=1)\
        corr_mean = mean_od - blank_od\
        surv = (corr_mean / corrected_control) * 100 if corrected_control > 0 else 0\
        stats.append([conc, mean_od, corr_mean, sd_od, (sd_od/mean_od)*100, surv])\
    \
    stats_df = pd.DataFrame(stats, columns=['Concentration', 'Mean OD', 'Corrected Mean', 'SD', '%CV', '% Survival'])\
    \
    # Fitting\
    fit_df = stats_df[stats_df['Concentration'] >= 0]\
    x, y = fit_df['Concentration'].values, fit_df['% Survival'].values\
    p0 = [0, 1.0, np.median(x[x>0]) if len(x[x>0]) else 1, 100]\
    \
    try:\
        popt, _ = curve_fit(four_PL, x, y, p0=p0, maxfev=10000)\
        ss_res = np.sum((y - four_PL(x, *popt))**2)\
        ss_tot = np.sum((y - np.mean(y))**2)\
        r2 = 1 - (ss_res / ss_tot)\
        return \{'df': stats_df, 'popt': popt, 'ic50': popt[2], 'r2': r2, 'success': True\}\
    except:\
        return \{'df': stats_df, 'success': False, 'ic50': None, 'r2': 0\}\
\
# --- UI Design ---\
st.title("
\f1 \uc0\u55358 \u56810 
\f0  Auto MTT Assay Analysis Tool")\
st.markdown("Web App 
\f2 \'ca\'d3\'cb\'c3\'d1\'ba\'c7\'d4\'e0\'a4\'c3\'d2\'d0\'cb\'ec
\f0  IC50, CC50 
\f2 \'e1\'c5\'d0
\f0  SI 
\f2 \'be\'c3\'e9\'cd\'c1\'a1\'c3\'d2\'bf
\f0  Publication Quality")\
\
with st.sidebar:\
    st.header("
\f1 \uc0\u9881 \u65039 
\f0  Settings")\
    blank_od = st.number_input("Blank OD Value", value=0.05, step=0.01)\
    theme = st.selectbox("Graph Theme", ["Standard (Red/Blue)", "Publication (Black/White)", "Pastel", "Nature"])\
    \
    st.divider()\
    st.markdown("### 
\f1 \uc0\u55357 \u56549 
\f0  Download Template")\
    template_df = create_template()\
    st.download_button(\
        label="Download Excel Template",\
        data=convert_df(template_df),\
        file_name="MTT_Template.csv",\
        mime="text/csv",\
    )\
\
col1, col2 = st.columns(2)\
\
data_normal = None\
data_cancer = None\
\
with col1:\
    st.subheader("
\f1 \uc0\u55357 \u57057 \u65039 
\f0  Normal Cells (for CC50)")\
    file_n = st.file_uploader("Upload CSV/Excel (Normal)", type=['csv', 'xlsx'])\
    if file_n:\
        if file_n.name.endswith('.csv'): data_normal = pd.read_csv(file_n)\
        else: data_normal = pd.read_excel(file_n)\
        st.success("Loaded Normal Cells Data")\
\
with col2:\
    st.subheader("
\f1 \uc0\u55358 \u56704 
\f0  Cancer/Target Cells (for IC50)")\
    file_c = st.file_uploader("Upload CSV/Excel (Cancer)", type=['csv', 'xlsx'])\
    if file_c:\
        if file_c.name.endswith('.csv'): data_cancer = pd.read_csv(file_c)\
        else: data_cancer = pd.read_excel(file_c)\
        st.success("Loaded Cancer Cells Data")\
\
if st.button("
\f1 \uc0\u55357 \u56960 
\f0  Analyze Data", type="primary"):\
    results = \{\}\
    if data_normal is not None:\
        results['normal'] = analyze_data(data_normal, blank_od)\
        results['normal']['name'] = "Normal Cells"\
    \
    if data_cancer is not None:\
        results['cancer'] = analyze_data(data_cancer, blank_od)\
        results['cancer']['name'] = "Cancer Cells"\
\
    # --- Result Display ---\
    if results:\
        st.divider()\
        st.header("
\f1 \uc0\u55357 \u56522 
\f0  Results")\
        \
        # Summary Metrics\
        cols = st.columns(3)\
        if 'cancer' in results and results['cancer']['success']:\
            cols[0].metric("IC50 (Target)", f"\{results['cancer']['ic50']:.4f\}", f"R2: \{results['cancer']['r2']:.2f\}")\
        \
        if 'normal' in results and results['normal']['success']:\
            cols[1].metric("CC50 (Normal)", f"\{results['normal']['ic50']:.4f\}", f"R2: \{results['normal']['r2']:.2f\}")\
            \
        if 'cancer' in results and 'normal' in results and results['cancer']['success'] and results['normal']['success']:\
            si = results['normal']['ic50'] / results['cancer']['ic50']\
            cols[2].metric("Selectivity Index (SI)", f"\{si:.4f\}", delta="Safe" if si > 3 else "Toxic")\
\
        # --- Plotting ---\
        st.subheader("
\f1 \uc0\u55357 \u56520 
\f0  Dose-Response Curve")\
        \
        # Theme Config\
        theme_cfg = \{\
            "Standard (Red/Blue)": \{'style': 'whitegrid', 'colors': ['#E63946', '#1D3557']\},\
            "Publication (Black/White)": \{'style': 'ticks', 'colors': ['black', '#666666']\},\
            "Pastel": \{'style': 'whitegrid', 'colors': ['#FF9AA2', '#85E3FF']\},\
            "Nature": \{'style': 'darkgrid', 'colors': ['#2A9D8F', '#F4A261']\}\
        \}\
        sel_theme = theme_cfg[theme]\
        sns.set_style(sel_theme['style'])\
        \
        fig, ax = plt.subplots(figsize=(8, 5))\
        colors = sel_theme['colors']\
        markers = ['o', 's']\
        \
        idx = 0\
        for key, res in results.items():\
            if not res['success']: continue\
            df = res['df']\
            mask = df['Concentration'] > 0\
            \
            c = colors[idx % 2]\
            \
            # Error bar\
            ax.errorbar(df.loc[mask, 'Concentration'], df.loc[mask, '% Survival'], \
                        yerr=df.loc[mask, 'SD'], fmt=markers[idx%2], color=c, capsize=4, label=f"\{res['name']\}")\
            \
            # Fit line\
            x_smooth = np.logspace(np.log10(df.loc[mask, 'Concentration'].min()/2), np.log10(df.loc[mask, 'Concentration'].max()*2), 100)\
            ax.plot(x_smooth, four_PL(x_smooth, *res['popt']), '-', color=c, alpha=0.8)\
            \
            # Star\
            ax.plot(res['ic50'], 50, '*', color='gold', markersize=15, markeredgecolor='black')\
            \
            idx += 1\
            \
        ax.set_xscale('log')\
        ax.axhline(50, color='gray', linestyle='--')\
        ax.set_xlabel('Concentration')\
        ax.set_ylabel('% Survival')\
        ax.set_ylim(-10, 120)\
        ax.legend()\
        \
        st.pyplot(fig)\
        \
        # Save Plot Button\
        buf = BytesIO()\
        fig.savefig(buf, format="png", dpi=300)\
        st.download_button("
\f1 \uc0\u55357 \u56510 
\f0  Download Graph (300 DPI)", data=buf.getvalue(), file_name="MTT_Plot.png", mime="image/png")\
        \
        # --- Data Tables ---\
        st.subheader("
\f1 \uc0\u55357 \u56523 
\f0  Detailed Data")\
        for key, res in results.items():\
            with st.expander(f"Show Data: \{res['name']\}"):\
                st.dataframe(res['df'])\
                st.download_button(f"Download CSV (\{res['name']\})", \
                                   data=convert_df(res['df']), \
                                   file_name=f"\{key\}_analysis.csv", \
                                   mime="text/csv")}