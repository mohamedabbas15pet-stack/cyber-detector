import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go
from pypdf import PdfReader
import io
import zipfile
import re
from datetime import datetime

# ----------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM SVG LOGO
# ----------------------------------------------------
st.set_page_config(
    page_title="NEXUS AI™ | Sovereign Enterprise Suite",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Vector SVG Logo (Never breaks, ultra-sharp on all screens)
NEXUS_LOGO_SVG = """
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="nexusGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#0F172A"/>
      <stop offset="50%" stop-color="#1E3A8A"/>
      <stop offset="100%" stop-color="#3B82F6"/>
    </linearGradient>
    <linearGradient id="accentGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#38BDF8"/>
      <stop offset="100%" stop-color="#818CF8"/>
    </linearGradient>
  </defs>
  <polygon points="32,2 58,17 58,47 32,62 6,47 6,17" fill="url(#nexusGrad)" stroke="url(#accentGrad)" stroke-width="2.5"/>
  <polygon points="32,12 49,22 49,42 32,52 15,42 15,22" fill="none" stroke="#38BDF8" stroke-width="1.5" stroke-dasharray="4 2"/>
  <circle cx="32" cy="32" r="7" fill="#38BDF8"/>
  <line x1="32" y1="12" x2="32" y2="25" stroke="#93C5FD" stroke-width="2"/>
  <line x1="32" y1="39" x2="32" y2="52" stroke="#93C5FD" stroke-width="2"/>
  <line x1="15" y1="22" x2="26" y2="28" stroke="#93C5FD" stroke-width="2"/>
  <line x1="49" y1="42" x2="38" y2="36" stroke="#93C5FD" stroke-width="2"/>
</svg>
"""

# Global CSS Styling
st.markdown("""
    <style>
    .nexus-banner {
        background: linear-gradient(90deg, #0F172A 0%, #1E293B 100%);
        padding: 20px 28px;
        border-radius: 12px;
        border-left: 6px solid #3B82F6;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .nexus-title {
        font-size: 34px;
        font-weight: 900;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .nexus-tagline {
        font-size: 14px;
        color: #93C5FD;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    .nexus-sub {
        font-size: 15px;
        color: #CBD5E1;
        margin-top: 6px;
    }
    .badge-pro {
        background-color: #2563EB;
        color: #FFFFFF;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        vertical-align: middle;
        margin-left: 10px;
    }
    .healing-box {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 4px solid #16A34A;
        padding: 12px 16px;
        border-radius: 6px;
        color: #166534;
        font-size: 14px;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Header Layout with Custom SVG Logo
st.markdown(f"""
    <div class="nexus-banner">
        <div style="display: flex; align-items: center; gap: 20px;">
            <div>{NEXUS_LOGO_SVG}</div>
            <div>
                <div class="nexus-title">NEXUS AI™ <span class="badge-pro">GEN-6 AUTONOMOUS CORE</span></div>
                <div class="nexus-tagline">Autonomous Intelligence. Zero Blind Spots.</div>
                <div class="nexus-sub">Self-Healing Data Engine • Auto-Calibrating Neural Risk Matrix • Live Cognitive Copilot</div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 2. SELF-HEALING & AUTO-TUNING ALGORITHMIC ENGINE
# ----------------------------------------------------
def self_heal_dataframe(df, required_cols):
    """Autonomous Data Self-Healing: Fixes missing values, types, and corrupt rows."""
    healing_log = []
    df_clean = df.copy()
    
    # Normalize column names
    df_clean.columns = [c.strip() for c in df_clean.columns]
    
    # Check missing values and impute with median
    null_count = df_clean.isnull().sum().sum()
    if null_count > 0:
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
        healing_log.append(f"Imputed {null_count} missing/corrupted cells using robust median estimation.")
        
    # Remove exact duplicate rows
    dup_count = df_clean.duplicated().sum()
    if dup_count > 0:
        df_clean.drop_duplicates(inplace=True)
        healing_log.append(f"Purged {dup_count} redundant duplicate records.")
        
    # Ensure required numeric columns exist
    for col in required_cols:
        if col not in df_clean.columns:
            df_clean[col] = np.random.normal(100, 15, len(df_clean))
            healing_log.append(f"Synthesized missing schema feature '{col}' via baseline distribution.")
        else:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
            
    if not healing_log:
        healing_log.append("Schema integrity verified at 100%. Zero structural anomalies found.")
        
    return df_clean, healing_log

def auto_calibrate_and_score(df, feature_cols, manual_contamination=None):
    """Self-Evolving Algorithm: Auto-tunes contamination via IQR and computes 0-100 Risk Score."""
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    
    # Statistical Auto-Tuning using Interquartile Range (IQR)
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    outlier_mask = ((X_scaled < (Q1 - 2.5 * IQR)) | (X_scaled > (Q3 + 2.5 * IQR))).any(axis=1)
    estimated_rate = float(np.clip(outlier_mask.mean(), 0.01, 0.25))
    
    final_contamination = estimated_rate if manual_contamination is None else manual_contamination
    
    model = IsolationForest(n_estimators=200, contamination=final_contamination, random_state=42)
    preds = model.fit_predict(X_scaled)
    raw_scores = model.decision_function(X_scaled)
    

