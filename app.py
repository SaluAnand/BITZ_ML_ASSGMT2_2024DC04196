"""
ML Assignment 2 - Streamlit Web Application
Breast Cancer Wisconsin Classification - Multiple ML Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report,
    roc_curve
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier | ML Assignment 2",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-title  { font-size:2.1rem; font-weight:700; color:#6a0572; text-align:center; margin-bottom:.2rem; }
.sub-title   { font-size:.95rem; color:#555; text-align:center; margin-bottom:1.5rem; }
.metric-card { background:#faf0ff; border-radius:10px; padding:12px 14px;
               text-align:center; border:1px solid #d9b0e8; }
.metric-value{ font-size:1.7rem; font-weight:700; color:#6a0572; }
.metric-label{ font-size:.75rem; color:#555; text-transform:uppercase; letter-spacing:.05em; }
.sec-hdr     { color:#6a0572; font-size:1.1rem; font-weight:600;
               border-bottom:2px solid #6a0572; padding-bottom:3px;
               margin-top:.8rem; margin-bottom:.7rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DISPLAY = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'knn':                 'K-Nearest Neighbor',
    'naive_bayes':         'Naïve Bayes (Gaussian)',
    'random_forest':       'Random Forest (Ensemble)',
    'xgboost':             'XGBoost (Ensemble)',
}

FEATURE_COLS = [
    'mean radius','mean texture','mean perimeter','mean area','mean smoothness',
    'mean compactness','mean concavity','mean concave points','mean symmetry',
    'mean fractal dimension','radius error','texture error','perimeter error',
    'area error','smoothness error','compactness error','concavity error',
    'concave points error','symmetry error','fractal dimension error',
    'worst radius','worst texture','worst perimeter','worst area','worst smoothness',
    'worst compactness','worst concavity','worst concave points','worst symmetry',
    'worst fractal dimension'
]

# ─────────────────────────────────────────────────────────────────────────────
# Load models & results
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    for key in MODEL_DISPLAY:
        try:
            with open(f'model/{key}.pkl', 'rb') as f:
                models[key] = pickle.load(f)
        except FileNotFoundError:
            models[key] = None
    return models

@st.cache_resource
def load_results():
    try:
        with open('model/model_results.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

all_models  = load_models()
all_results = load_results()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=130)
    except Exception:
        st.markdown("**BITS Pilani WILP**")
    st.markdown("### ML Assignment 2 - 2024DC04196")
    st.divider()
    st.markdown("### 🔧 Model Selection")
    selected_model_key = st.selectbox(
        "Choose ML Model",
        options=list(MODEL_DISPLAY.keys()),
        format_func=lambda k: MODEL_DISPLAY[k],
        index=0
    )
    st.divider()
    st.markdown("### 📂 Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload CSV (test split only)",
        type=['csv'],
        help="Upload test_data.csv generated by the training script"
    )
    st.divider()
    st.markdown(
        "**Dataset:** Breast Cancer Wisconsin  \n"
        "**Task:** Binary Classification  \n"
        "**Features:** 30  \n"
        "**Instances:** 569  \n"
        "**Source:** sklearn / UCI"
    )

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🎗️ Breast Cancer Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">ML Assignment 2 · BITS Pilani WILP · M.Tech AIML/DSE · '
    'Breast Cancer Wisconsin (Diagnostic) Dataset</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Model Evaluation", "📈 Model Comparison", "🔮 Predict", "ℹ️ About"]
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def compute_from_df(df_in, model_key):
    model = all_models.get(model_key)
    if model is None:
        return None
    missing = [c for c in FEATURE_COLS if c not in df_in.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return None
    X    = df_in[FEATURE_COLS]
    y_tr = df_in['target'].values if 'target' in df_in.columns else None
    y_pd = model.predict(X)
    y_pb = model.predict_proba(X)[:, 1]
    if y_tr is not None:
        return {
            'accuracy':  round(accuracy_score(y_tr, y_pd),  4),
            'auc':       round(roc_auc_score(y_tr, y_pb),   4),
            'precision': round(precision_score(y_tr, y_pd), 4),
            'recall':    round(recall_score(y_tr, y_pd),    4),
            'f1':        round(f1_score(y_tr, y_pd),        4),
            'mcc':       round(matthews_corrcoef(y_tr, y_pd), 4),
            'confusion_matrix': confusion_matrix(y_tr, y_pd).tolist(),
            'classification_report': classification_report(
                y_tr, y_pd, target_names=['Malignant','Benign']),
            'y_true': y_tr, 'y_prob': y_pb,
        }
    return {'predictions': y_pd, 'probabilities': y_pb}

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Evaluation
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        f'<div class="sec-hdr">Selected Model: {MODEL_DISPLAY[selected_model_key]}</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        st.success(f"✅ **{uploaded_file.name}** — {df_up.shape[0]} rows × {df_up.shape[1]} cols")
        metrics = compute_from_df(df_up, selected_model_key)
    elif all_results:
        metrics = all_results.get(selected_model_key)
        st.info("ℹ️ Showing pre-computed training results. Upload **test_data.csv** for live evaluation.")
    else:
        metrics = None
        st.warning("⚠️ Models not found. Run `model/train_models.py` first.")

    if metrics and 'accuracy' in metrics:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, label, val in [
            (c1, "Accuracy",  metrics['accuracy']),
            (c2, "AUC Score", metrics['auc']),
            (c3, "Precision", metrics['precision']),
            (c4, "Recall",    metrics['recall']),
            (c5, "F1 Score",  metrics['f1']),
            (c6, "MCC Score", metrics['mcc']),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{val:.4f}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.markdown('<div class="sec-hdr">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = np.array(metrics['confusion_matrix'])
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                        xticklabels=['Malignant','Benign'],
                        yticklabels=['Malignant','Benign'],
                        ax=ax, linewidths=0.5)
            ax.set_xlabel('Predicted', fontsize=10)
            ax.set_ylabel('Actual', fontsize=10)
            ax.set_title(f'Confusion Matrix\n{MODEL_DISPLAY[selected_model_key]}', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

        with col_roc:
            st.markdown('<div class="sec-hdr">ROC Curve</div>', unsafe_allow_html=True)
            if 'y_true' in metrics and 'y_prob' in metrics:
                fpr, tpr, _ = roc_curve(metrics['y_true'], metrics['y_prob'])
            else:
                fpr = np.linspace(0, 1, 200)
                auc_v = metrics['auc']
                tpr   = np.clip(fpr + (auc_v - 0.5)*2*np.sqrt(fpr*(1-fpr)), 0, 1)
            fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
            ax2.plot(fpr, tpr, color='#6a0572', lw=2, label=f'AUC = {metrics["auc"]:.4f}')
            ax2.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
            ax2.set_xlim([0,1]); ax2.set_ylim([0,1.02])
            ax2.set_xlabel('False Positive Rate', fontsize=10)
            ax2.set_ylabel('True Positive Rate', fontsize=10)
            ax2.set_title(f'ROC Curve\n{MODEL_DISPLAY[selected_model_key]}', fontsize=10)
            ax2.legend(loc='lower right', fontsize=9)
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig2); plt.close(fig2)

        st.markdown('<div class="sec-hdr">Classification Report</div>', unsafe_allow_html=True)
        st.code(metrics['classification_report'], language='text')

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Comparison
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-hdr">All 6 Models — Evaluation Metrics</div>', unsafe_allow_html=True)
    if all_results:
        rows = []
        for key, m in all_results.items():
            rows.append({
                'Model': MODEL_DISPLAY[key],
                'Accuracy': m['accuracy'], 'AUC': m['auc'],
                'Precision': m['precision'], 'Recall': m['recall'],
                'F1 Score': m['f1'], 'MCC': m['mcc'],
            })
        cdf = pd.DataFrame(rows).set_index('Model')
        st.dataframe(cdf.style.background_gradient(cmap='Purples', axis=0).format("{:.4f}"),
                     use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Visual Comparison</div>', unsafe_allow_html=True)
        metric_pick = st.selectbox("Select metric",
            ['Accuracy','AUC','Precision','Recall','F1 Score','MCC'])

        fig3, ax3 = plt.subplots(figsize=(9, 4))
        colors = ['#6a0572' if v == cdf[metric_pick].max()
                  else '#d9b0e8' for v in cdf[metric_pick]]
        bars = ax3.barh(cdf.index, cdf[metric_pick], color=colors,
                        edgecolor='white', height=0.55)
        ax3.set_xlim(0, 1.1)
        ax3.set_xlabel(metric_pick, fontsize=10)
        ax3.set_title(f'Model Comparison — {metric_pick}', fontsize=12)
        for bar, val in zip(bars, cdf[metric_pick]):
            ax3.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
                     f'{val:.4f}', va='center', fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig3); plt.close(fig3)

        # Radar chart
        st.markdown('<div class="sec-hdr">Radar Chart — All Metrics</div>', unsafe_allow_html=True)
        mets = ['Accuracy','AUC','Precision','Recall','F1 Score','MCC']
        N = len(mets)
        angles = [n/float(N)*2*np.pi for n in range(N)] + [0]
        fig4, ax4 = plt.subplots(figsize=(7,5.5), subplot_kw=dict(polar=True))
        palette = ['#6a0572','#9b59b6','#c39bd3','#e74c3c','#27ae60','#2e86c1']
        for i,(key,m) in enumerate(all_results.items()):
            vals = [m['accuracy'],m['auc'],m['precision'],
                    m['recall'],m['f1'],(m['mcc']+1)/2]
            ax4.plot(angles, vals+[vals[0]], 'o-', lw=1.8,
                     label=MODEL_DISPLAY[key], color=palette[i], markersize=4)
            ax4.fill(angles, vals+[vals[0]], alpha=0.07, color=palette[i])
        ax4.set_xticks(angles[:-1]); ax4.set_xticklabels(mets, fontsize=9)
        ax4.set_ylim(0,1)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.35,1.1), fontsize=8)
        ax4.set_title('Model Performance Radar\n(MCC normalised to [0,1])', fontsize=10, pad=14)
        plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)
    else:
        st.warning("Run `model/train_models.py` to generate results.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="sec-hdr">🔮 Single-Instance Prediction</div>', unsafe_allow_html=True)
    st.markdown("Enter cell nucleus measurements to predict **Malignant** or **Benign**.")
    model = all_models.get(selected_model_key)
    if model is None:
        st.warning("Model not loaded. Run the training script first.")
    else:
        # Representative mean values from the dataset
        defaults = {
            'mean radius':14.1,'mean texture':19.3,'mean perimeter':91.97,'mean area':654.9,
            'mean smoothness':0.0964,'mean compactness':0.1044,'mean concavity':0.0888,
            'mean concave points':0.0489,'mean symmetry':0.1812,'mean fractal dimension':0.0628,
            'radius error':0.405,'texture error':1.217,'perimeter error':2.866,'area error':40.34,
            'smoothness error':0.00704,'compactness error':0.02565,'concavity error':0.03192,
            'concave points error':0.01180,'symmetry error':0.02054,'fractal dimension error':0.003795,
            'worst radius':16.27,'worst texture':25.68,'worst perimeter':107.26,'worst area':880.6,
            'worst smoothness':0.1323,'worst compactness':0.2543,'worst concavity':0.2722,
            'worst concave points':0.1142,'worst symmetry':0.2906,'worst fractal dimension':0.08490
        }
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURE_COLS):
            with cols[i % 3]:
                inputs[feat] = st.number_input(feat, value=float(defaults[feat]),
                                               format="%.5f", key=feat)

        if st.button("🎗️ Predict", type='primary', use_container_width=True):
            inp = pd.DataFrame([inputs])
            pred = model.predict(inp)[0]
            prob = model.predict_proba(inp)[0]
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                if pred == 0:
                    st.error(f"### ⚠️ Malignant\nConfidence: **{prob[0]:.1%}**")
                else:
                    st.success(f"### ✅ Benign\nConfidence: **{prob[1]:.1%}**")
            with col_r2:
                fig5, ax5 = plt.subplots(figsize=(4,3))
                ax5.barh(['Malignant','Benign'], prob,
                          color=['#e74c3c','#27ae60'], height=0.45)
                ax5.set_xlim(0,1)
                ax5.set_xlabel('Probability', fontsize=10)
                ax5.set_title(f'Prediction Confidence\n({MODEL_DISPLAY[selected_model_key]})', fontsize=9)
                for j,v in enumerate(prob):
                    ax5.text(v+0.01, j, f'{v:.1%}', va='center', fontsize=9)
                ax5.grid(axis='x', alpha=0.3)
                plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – About
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("""
### About This App
Demonstrates end-to-end ML deployment for **binary classification** using the
Breast Cancer Wisconsin (Diagnostic) dataset.

#### Dataset
- **Source:** UCI Machine Learning Repository / sklearn
- **Instances:** 569 | **Features:** 30 | **Task:** Binary (Malignant=0 / Benign=1)

#### Models Implemented
| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | Linear |
| 2 | Decision Tree | Tree-based |
| 3 | K-Nearest Neighbor | Instance-based |
| 4 | Naïve Bayes (Gaussian) | Probabilistic |
| 5 | Random Forest | Ensemble (Bagging) |
| 6 | XGBoost | Ensemble (Boosting) |

#### App Features
- 📂 Upload test CSV for live evaluation
- 🔽 Model selection dropdown
- 📊 All 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- 📉 Confusion matrix + ROC curve
- 📋 Classification report
- 📈 Model comparison table + bar + radar charts
- 🔮 Single-instance prediction with probability display
    """)

st.divider()
st.markdown(
    "<div style='text-align:center;color:#888;font-size:.8rem;'>"
    "ML Assignment 2 · BITS Pilani WILP · M.Tech AIML/DSE · "
    "Breast Cancer Wisconsin Classification</div>",
    unsafe_allow_html=True
)
