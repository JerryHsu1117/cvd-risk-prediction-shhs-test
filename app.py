"""
app.py — CVD Risk Prediction Streamlit App

Run from the project root:
    streamlit run src/app.py

Input fields are determined by SHAP feature importance (Output/shap_top_features.json).
Top 5 sleep metrics on the left, top 5 health indicators on the right.
Remaining features are filled with population medians.
"""

import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

# ── Paths (run from project root) ─────────────────────────────────────────────
ROOT       = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(ROOT, 'Output', 'model.pkl')
SHAP_PATH  = os.path.join(ROOT, 'Output', 'shap_top_features.json')

THRESHOLD = 0.55

CATEGORICAL_COLS = [
    'sex', 'race', 'smoking_status', 'self_reported_hypertension',
    'history_of_diabetes', 'taking_bp_medication',
]

ALL_FEATURES = [
    'apnea_events_per_hour', 'avg_oxygen_saturation', 'min_oxygen_saturation',
    'sleep_efficiency_pct', 'wake_after_sleep_onset_min', 'bmi',
    'waist_circumference_cm', 'systolic_bp', 'diastolic_bp', 'cholesterol',
    'hdl_cholesterol', 'self_reported_hypertension', 'history_of_diabetes',
    'taking_bp_medication', 'age_at_baseline', 'sex', 'race', 'smoking_status',
]

# Population medians — used for features not shown in the form
DEFAULTS = {
    'bmi':                       27.6,
    'waist_circumference_cm':    97.0,
    'systolic_bp':               124.0,
    'diastolic_bp':              72.0,
    'cholesterol':               204.0,
    'self_reported_hypertension': 0.0,
    'history_of_diabetes':        0.0,
    'race':                       1,
}

# Widget config per feature
WIDGET = {
    'apnea_events_per_hour':      {'label': 'Apnea Events Per Hour (AHI)',       'type': 'float',  'min': 0.0,  'max': 50.0,  'default': 10.0, 'step': 0.1},
    'min_oxygen_saturation':      {'label': 'Minimum Oxygen Saturation (%)',      'type': 'float',  'min': 70.0, 'max': 100.0, 'default': 87.0, 'step': 0.1},
    'sleep_efficiency_pct':       {'label': 'Sleep Efficiency (%)',               'type': 'float',  'min': 50.0, 'max': 100.0, 'default': 85.0, 'step': 0.1},
    'avg_oxygen_saturation':      {'label': 'Avg Oxygen Saturation (%)',          'type': 'float',  'min': 85.0, 'max': 100.0, 'default': 94.5, 'step': 0.1},
    'wake_after_sleep_onset_min': {'label': 'Wake After Sleep Onset (min)',       'type': 'float',  'min': 0.0,  'max': 180.0, 'default': 50.0, 'step': 1.0},
    'age_at_baseline':            {'label': 'Age (years)',                        'type': 'int',    'min': 18,   'max': 120,   'default': 64},
    'sex':                        {'label': 'Sex',                                'type': 'select', 'options': {'Male': 1, 'Female': 2},                      'default': 'Male'},
    'taking_bp_medication':       {'label': 'Taking Blood Pressure Medication',   'type': 'select', 'options': {'No': 0.0, 'Yes': 1.0},                       'default': 'No'},
    'smoking_status':             {'label': 'Smoking Status',                     'type': 'select', 'options': {'Never': 0.0, 'Former': 1.0, 'Current': 2.0}, 'default': 'Never'},
    'hdl_cholesterol':            {'label': 'HDL Cholesterol (mg/dL)',            'type': 'float',  'min': 10.0, 'max': 100.0, 'default': 48.0, 'step': 1.0},
}


@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)

    with open(SHAP_PATH) as f:
        shap_data = json.load(f)

    preprocessor = model.named_steps['preprocessor']
    lr_model     = model.named_steps['model']

    # Feature names from the fitted preprocessor — no dataset CSV needed
    numeric_cols   = [c for c in ALL_FEATURES if c not in CATEGORICAL_COLS]
    ohe_names      = (preprocessor.named_transformers_['cat']['encoder']
                      .get_feature_names_out(CATEGORICAL_COLS).tolist())
    all_feat_names = numeric_cols + ohe_names

    # Zero background: StandardScaler normalizes numerics to mean=0, so zeros equal
    # the population mean — the correct SHAP baseline. OHE features are an approximation
    # but directionally accurate for visualization.
    background = np.zeros((1, len(all_feat_names)))
    explainer  = shap.LinearExplainer(lr_model, background)

    return model, shap_data, explainer, preprocessor, all_feat_names


def parent_feature(col_name):
    for cat in CATEGORICAL_COLS:
        if col_name.startswith(cat + '_'):
            return cat
    return col_name


def build_input_row(user_inputs):
    row = DEFAULTS.copy()
    row.update(user_inputs)
    return pd.DataFrame([{f: row[f] for f in ALL_FEATURES}])


def compute_shap_contributions(model, explainer, preprocessor, all_feat_names, X_input):
    X_t       = preprocessor.transform(X_input)
    shap_vals = explainer.shap_values(X_t)[0]
    raw       = dict(zip(all_feat_names, shap_vals))
    # Collapse OHE columns back to original feature names
    aggregated = {}
    for col, val in raw.items():
        parent = parent_feature(col)
        aggregated[parent] = aggregated.get(parent, 0.0) + val
    return aggregated


def render_shap_chart(ax, features, contributions, title):
    vals   = [contributions.get(f, 0.0) for f in features]
    colors = ['#e74c3c' if v > 0 else '#2ecc71' for v in vals]
    ax.barh(features[::-1], vals[::-1], color=colors[::-1], edgecolor='white')
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('SHAP value  (red = increases risk, green = decreases risk)', fontsize=9)
    ax.tick_params(axis='y', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Page ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='CVD Risk Prediction', layout='wide')
st.title('CVD Risk Prediction')
st.write('Enter sleep and health measurements to assess cardiovascular disease risk.')

model, shap_data, explainer, preprocessor, all_feat_names = load_resources()
TOP_5_SLEEP  = shap_data['TOP_5_SLEEP']
TOP_5_HEALTH = shap_data['TOP_5_HEALTH']

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader('Patient Inputs')
col_sleep, col_health = st.columns(2)
user_inputs = {}


def render_widget(container, feat):
    cfg = WIDGET[feat]
    with container:
        if cfg['type'] == 'float':
            return st.number_input(cfg['label'], min_value=float(cfg['min']),
                                   max_value=float(cfg['max']), value=float(cfg['default']),
                                   step=float(cfg['step']), key=feat)
        elif cfg['type'] == 'int':
            return st.number_input(cfg['label'], min_value=int(cfg['min']),
                                   max_value=int(cfg['max']), value=int(cfg['default']),
                                   step=1, key=feat)
        elif cfg['type'] == 'select':
            opts    = list(cfg['options'].keys())
            default = opts.index(cfg['default'])
            sel     = st.selectbox(cfg['label'], opts, index=default, key=feat)
            return cfg['options'][sel]


with col_sleep:
    st.write('**Sleep Metrics**')
    for feat in TOP_5_SLEEP:
        user_inputs[feat] = render_widget(col_sleep, feat)

with col_health:
    st.write('**Health Indicators**')
    for feat in TOP_5_HEALTH:
        user_inputs[feat] = render_widget(col_health, feat)

# ── Predict button ────────────────────────────────────────────────────────────
if st.button('Predict CVD Risk', type='primary', use_container_width=True):

    X_input = build_input_row(user_inputs)
    prob    = model.predict_proba(X_input)[0, 1]

    st.divider()
    st.subheader('Prediction Result')

    if prob >= 0.70:
        st.error(f'**HIGH RISK** — {prob * 100:.1f}% CVD probability')
        explanation = 'Multiple high-risk factors are elevated. Speak with a physician.'
    elif prob >= 0.45:
        st.warning(f'**MEDIUM RISK** — {prob * 100:.1f}% CVD probability')
        explanation = 'Some risk factors are above normal. Consider a cardiovascular checkup.'
    else:
        st.success(f'**LOW RISK** — {prob * 100:.1f}% CVD probability')
        explanation = 'Current indicators suggest low CVD risk based on these measurements.'

    st.write(explanation)

    # SHAP contributions chart
    contributions = compute_shap_contributions(
        model, explainer, preprocessor, all_feat_names, X_input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    render_shap_chart(ax1, TOP_5_SLEEP,  contributions, 'Sleep Metrics — Individual Contribution')
    render_shap_chart(ax2, TOP_5_HEALTH, contributions, 'Health Indicators — Individual Contribution')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        'For educational purposes only. Not a substitute for medical advice '
        'or clinical diagnosis. Built for BSAN6070 — LMU, Spring 2026.'
    )
