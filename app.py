import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from io import BytesIO

# ==========================================
# 1. SETUP & CONSTANTS (ประกาศตัวแปรคงที่)
# ==========================================
st.set_page_config(page_title="MTT Analysis Pro", page_icon="🧪", layout="wide")

# กำหนดตัวแปรสำหรับกราฟไว้ตรงนี้ (Global) เพื่อป้องกัน NameError
MARKER_MAP = {
    "Star (*)": '*', 
    "Diamond (D)": 'D', 
    "Circle (o)": 'o', 
    "Cross (X)": 'X', 
    "Triangle (^)": '^'
}
POINT_MARKERS = ['o', 's']  # จุดข้อมูล (วงกลม, สี่เหลี่ยม)

# --- Core Math Functions ---
def four_PL(x, A, B, C, D):
    return D + (A - D) / (1.0 + (x / C)**B)

@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

def create_template():
    """สร้าง Template DataFrame"""
    return pd.DataFrame({
        'Concentration': [0, 1, 10, 100],
        'Rep1': [1.0, 0.9, 0.5, 0.1],
        'Rep2': [1.02, 0.88, 0.48, 0.12],
        'Rep3': [0.98, 0.92, 0.52, 0.09]
    })

def parse_manual_input(text_input):
    """แปลงข้อความ Manual Input เป็น DataFrame"""
    try:
        data = []
        lines = text_input.strip().split('\n')
        for line in lines:
            if not line.strip(): continue
            parts = [float(x.strip()) for x in line.split(',')]
            data.append(parts)
        if not data: return None
        n_reps = len(data[0]) - 1
        cols = ['Concentration'] + [f'Rep{i+1}' for i in range(n_reps)]
        return pd.DataFrame(data, columns=cols)
    except:
        return None

def generate_mock_data(is_normal=False):
    """สร้างข้อมูลจำลอง"""
    ic50_target = 15
    concs = np.array([0, 0.1, 1, 5, 10, 50, 100, 500])
    actual_ic50 = ic50_target * 10 if is_normal else ic50_target
    def sim_od(c, center):
        if c == 0: base_od = 1.0 
        else: 
            viability = four_PL(c, 0, 1.2, center, 100)
            base_od = (viability / 100) * 1.0
        return np.random.normal(base_od, 0.05 * base_od, 3) + 0.05
    data = [sim_od(c, actual_ic50) for c in concs]
    cols = ['Concentration'] + [f'Rep{i+1}' for i in range(3)]
    df = pd.DataFrame(columns=cols)
    df['Concentration'] = concs
    for i in range(3): df[f'Rep{i+1}'] = [row[i] for row in data]
    return df

def analyze_data(df, blank_od):
    """ฟังก์ชันคำนวณและ Fit Curve"""
    control_row = df[df['Concentration'] == 0]
    raw_control = df.iloc[:, 1:].max().mean() if control_row.empty else control_row.iloc[:, 1:].values.mean()
    corrected_control = raw_control - blank_od
    
    stats = []
    for _, row in df.iterrows():
        conc = row['Concentration']
        reps = row.iloc[1:].values.astype(float)
        mean_od = np.nanmean(reps)
        sd_od = np.nanstd(reps, ddof=1)
        corr_mean = mean_od - blank_od
        surv = (corr_mean / corrected_control) * 100 if corrected_control > 0 else 0
        stats.append([conc, mean_od, corr_mean, sd_od, (sd_od/mean_od)*100, surv])
    
    stats_df = pd.DataFrame(stats, columns=['Concentration', 'Mean OD', 'Corrected Mean', 'SD', '%CV', '% Survival'])
    
    fit_df = stats_df[stats_df['Concentration'] >= 0]
    x, y = fit_df['Concentration'].values, fit_df['% Survival'].values
    p0 = [0, 1.0, np.median(x[x>0]) if len(x[x>0]) else 1, 100]
    
    try:
        popt, _ = curve_fit(four_PL, x, y, p0=p0, maxfev=10000)
        ss_res = np.sum((y - four_PL(x, *popt))**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        return {'df': stats_df, 'popt': popt, 'ic50': popt[2], 'r2': r2, 'success': True}
    except:
        return {'df': stats_df, 'success': False, 'ic50': None, 'r2': 0}

# ==========================================
# 2. SIDEBAR SETTINGS
# ==========================================
st.title("🧪 MTT Analysis: Custom Graphing")
st.markdown("วิเคราะห์ IC50/CC50 พร้อมระบุชื่อเซลล์และปรับแต่งกราฟได้ตามต้องการ")

with st.sidebar:
    st.header("📥 Tools & Settings")
    
    # Template Download
    template_df = create_template()
    st.download_button(
        label="📄 Download Data Template",
        data=convert_df(template_df),
        file_name="MTT_Template.csv",
        mime="text/csv",
        help="ดาวน์โหลดไฟล์ CSV ต้นแบบสำหรับกรอกข้อมูล"
    )
    st.divider()
    
    blank_od = st.number_input("Blank OD Value", value=0.05, step=0.01)
    
    st.divider()
    st.header("🎨 Graph Customization")
    
    theme_choice = st.selectbox(
        "Color Theme", 
        ["Standard (Red/Blue)", "Publication (Black/White)", "Pastel (Soft)", "Nature (Green/Orange)"]
    )
    
    marker_choice = st.selectbox(
        "IC50/CC50 Marker Shape", 
        list(MARKER_MAP.keys())
    )

# ==========================================
# 3. INPUT SECTION
# ==========================================

# เลือกโหมด
analysis_mode = st.radio(
    "1️⃣ เลือกโหมดการวิเคราะห์:",
    ["IC50 Only (Target Cells)", "CC50 Only (Normal Cells)", "Both (Calculate SI)"],
    horizontal=True
)

st.divider()
st.markdown("### 2️⃣ นำเข้าข้อมูลและตั้งชื่อเซลล์")

def render_input_box(label, key_prefix, default_mock_normal=False):
    """สร้างกล่องรับข้อมูลพร้อมช่องตั้งชื่อ"""
    with st.container(border=True):
        st.markdown(f"**ข้อมูล: {label}**")
        
        custom_name = st.text_input(f"🏷️ ระบุชื่อเซลล์ (จะแสดงในกราฟ)", value=label, key=f"{key_prefix}_name")
        
        method = st.radio(
            f"วิธีนำเข้าข้อมูล:",
            ["📂 Upload Excel/CSV", "⌨️ Manual Input", "🎲 Demo Data"],
            key=f"{key_prefix}_method",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        df_out = None
        
        if method == "📂 Upload Excel/CSV":
            f = st.file_uploader(f"เลือกไฟล์ ({label})", type=['csv', 'xlsx'], key=f"{key_prefix}_file")
            if f:
                if f.name.endswith('.csv'): df_out = pd.read_csv(f)
                else: df_out = pd.read_excel(f)
                if 'Concentration' not in df_out.columns:
                     df_out.rename(columns={df_out.columns[0]: 'Concentration'}, inplace=True)
                st.success(f"✅ Loaded {f.name}")
                
        elif method == "⌨️ Manual Input":
            st.caption("Format: Conc, Rep1, Rep2... (0 สำหรับ Control)")
            txt = st.text_area("วางข้อมูลที่นี่", height=100, key=f"{key_prefix}_txt", placeholder="0, 1.2, 1.1\n10, 0.8, 0.9")
            if txt:
                df_out = parse_manual_input(txt)
                if df_out is not None: st.success("✅ Data Parsed")
                else: st.error("❌ Invalid Format")

        elif method == "🎲 Demo Data":
            df_out = generate_mock_data(is_normal=default_mock_normal)
            st.info("✅ Mock Data Loaded")
            
        return df_out, custom_name

col1, col2 = st.columns(2)
data_cancer, data_normal = None, None
c_name, n_name = "Target Cells", "Normal Cells"

if analysis_mode == "IC50 Only (Target Cells)":
    with col1: data_cancer, c_name = render_input_box("Target Cells", "cancer", False)

elif analysis_mode == "CC50 Only (Normal Cells)":
    with col1: data_normal, n_name = render_input_box("Normal Cells", "normal", True)

elif analysis_mode == "Both (Calculate SI)":
    with col1: data_normal, n_name = render_input_box("Normal Cells", "normal", True)
    with col2: data_cancer, c_name = render_input_box("Target Cells", "cancer", False)

# ==========================================
# 4. ANALYSIS & PLOTTING
# ==========================================
ready = False
if analysis_mode == "IC50 Only (Target Cells)" and data_cancer is not None: ready = True
elif analysis_mode == "CC50 Only (Normal Cells)" and data_normal is not None: ready = True
elif analysis_mode == "Both (Calculate SI)" and data_cancer is not None and data_normal is not None: ready = True

if ready:
    st.divider()
    if st.button("🚀 Run Analysis & Plot", type="primary", use_container_width=True):
        results = {}
        
        if data_cancer is not None:
            res = analyze_data(data_cancer, blank_od)
            res['name'] = c_name
            results['cancer'] = res
            
        if data_normal is not None:
            res = analyze_data(data_normal, blank_od)
            res['name'] = n_name
            results['normal'] = res

        # Display Metrics
        st.markdown("### 📊 ผลลัพธ์ (Results)")
        m_cols = st.columns(3)
        if 'cancer' in results and results['cancer']['success']:
            m_cols[0].metric(f"IC50 ({c_name})", f"{results['cancer']['ic50']:.4f}", f"R²: {results['cancer']['r2']:.2f}")
        
        if 'normal' in results and results['normal']['success']:
            m_cols[1].metric(f"CC50 ({n_name})", f"{results['normal']['ic50']:.4f}", f"R²: {results['normal']['r2']:.2f}")
            
        if 'cancer' in results and 'normal' in results and results['cancer']['success'] and results['normal']['success']:
            si = results['normal']['ic50'] / results['cancer']['ic50']
            m_cols[2].metric("Selectivity Index (SI)", f"{si:.4f}", delta="Safe (>3)" if si > 3 else "Toxic")

        # --- PLOTTING ---
        st.markdown("### 📈 Dose-Response Curve")
        
        # Theme Setup
        theme_cfg = {
            "Standard (Red/Blue)": {'style': 'whitegrid', 'colors': ['#E63946', '#1D3557'], 'bg': 'white'},
            "Publication (Black/White)": {'style': 'ticks', 'colors': ['black', '#666666'], 'bg': 'white'},
            "Pastel (Soft)": {'style': 'whitegrid', 'colors': ['#FF9AA2', '#85E3FF'], 'bg': '#FAFAFA'},
            "Nature (Green/Orange)": {'style': 'darkgrid', 'colors': ['#2A9D8F', '#F4A261'], 'bg': '#F0F0F0'}
        }
        selected_theme = theme_cfg[theme_choice]
        sns.set_style(selected_theme['style'])
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(selected_theme['bg'])
        ax.set_facecolor(selected_theme['bg'])
        
        colors = selected_theme['colors']
        
        # ดึง Marker ที่เลือกมาจากตัวแปร Global
        ic50_marker_symbol = MARKER_MAP[marker_choice]
        
        idx = 0
        for key, res in results.items():
            if not res['success']: continue
            df = res['df']
            mask = df['Concentration'] > 0
            
            c = colors[idx % 2]
            
            # Use Global POINT_MARKERS (แก้ปัญหา NameError)
            pt_marker = POINT_MARKERS[idx % 2]
            
            # Plot Data
            ax.errorbar(df.loc[mask, 'Concentration'], df.loc[mask, '% Survival'], 
                        yerr=df.loc[mask, 'SD'], fmt=pt_marker, color=c, 
                        capsize=4, label=f"{res['name']} Data", alpha=0.7)
            
            # Plot Fit Line
            x_smooth = np.logspace(np.log10(df.loc[mask, 'Concentration'].min()/2), np.log10(df.loc[mask, 'Concentration'].max()*2), 100)
            val_label = f"{res['ic50']:.2f}"
            ax.plot(x_smooth, four_PL(x_smooth, *res['popt']), '-', color=c, linewidth=2, label=f"{res['name']} Fit ({val_label})")
            
            # Plot IC50 Point
            ax.plot(res['ic50'], 50, marker=ic50_marker_symbol, color='gold', markersize=12, markeredgecolor='black', zorder=10)
            
            idx += 1
            
        ax.set_xscale('log')
        ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Concentration')
        ax.set_ylabel('% Cell Survival')
        ax.set_ylim(-10, 120)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        ax.grid(True, which='major', alpha=0.3)
        
        col_plot, col_dl = st.columns([3, 1])
        with col_plot: st.pyplot(fig)
        with col_dl:
            st.markdown("<br><br>", unsafe_allow_html=True)
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button("💾 Download Graph (300 DPI)", data=buf.getvalue(), file_name="MTT_Custom_Graph.png", mime="image/png")
            
        st.divider()
        st.markdown("### 📋 Download Report")
        for key, res in results.items():
            with st.expander(f"View Data: {res['name']}"):
                st.dataframe(res['df'])
                st.download_button(f"Download CSV ({res['name']})", data=convert_df(res['df']), file_name=f"{res['name']}_data.csv")

else:
    st.info("👈 กรุณากรอกข้อมูลให้ครบถ้วนเพื่อเริ่มวิเคราะห์")
