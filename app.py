from __future__ import annotations

import json
import pickle
import warnings as _warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Suppress sklearn version-mismatch noise — models still predict correctly
_warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*", category=UserWarning)
_warnings.filterwarnings("ignore", message="Trying to unpickle estimator",   category=UserWarning)

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# Constants
# ============================================================
APP_DIR   = Path(__file__).parent
DATA_DIR  = APP_DIR / "data"
REGISTRY_PATH = APP_DIR / "model_registry.json"
PARETO_PATH   = DATA_DIR / "combined_pareto_front.csv"

SCM_COLUMNS   = ["Quartz", "Fly Ash", "Bagasse", "Silica Fume", "Calcium Carbonate"]
FEATURE_COLUMNS = ["Water", "Cement"] + SCM_COLUMNS + ["Fiber"]
SCM_TYPES     = SCM_COLUMNS.copy()

PRIMARY_MODEL_NAME = "Random Forest Tuned"

TRAINING_RANGES: Dict[str, Dict[str, Any]] = {
    "Water":      {"min": 300.0, "max": 400.0,  "unit": "kg/m³", "reference": 350.0},
    "Cement":     {"min": 750.0, "max": 1000.0, "unit": "kg/m³", "reference": 875.0},
    "SCM_Amount": {"min": 0.0,   "max": 250.0,  "unit": "kg/m³", "reference": 125.0},
    "Fiber":      {"min": 0.0,   "max": 0.30,   "unit": "%",     "reference": 0.15},
}

DEFAULT_INPUTS: Dict[str, Any] = {
    "Water": 350.0, "Cement": 825.0,
    "SCM_Type": "Quartz", "SCM_Amount": 125.0, "Fiber": 0.15,
}

SAMPLE_MIXES: Dict[str, Dict[str, Any]] = {
    "Balanced Mix":        {"Water": 350.0, "Cement": 825.0, "SCM_Type": "Quartz",      "SCM_Amount": 125.0, "Fiber": 0.15},
    "Low Cement Mix":      {"Water": 340.0, "Cement": 760.0, "SCM_Type": "Fly Ash",     "SCM_Amount": 180.0, "Fiber": 0.18},
    "High Strength Trial": {"Water": 320.0, "Cement": 950.0, "SCM_Type": "Silica Fume", "SCM_Amount": 120.0, "Fiber": 0.25},
}

RELIABLE_MODEL_TYPES = {
    "Random Forest", "XGBoost", "LightGBM",
    "CatBoost", "Gradient Boosting", "Decision Tree",
}

STRENGTH_COLUMN_CANDIDATES = [
    "Predicted_Strength", "Predicted Strength", "Prediction",
    "Prediction (MPa)", "Tensile_Strength", "Tensile Strength", "Strength", "target",
]

_PLOTLY_TEMPLATE = "plotly_white"

# ============================================================
# Page config
# ============================================================
st.set_page_config(
    page_title="TensAile Lab",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS — refined engineering-grade design
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700;800&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.block-container {
    padding-top: 5rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1520px;
}
[data-testid="stHeader"] {
    background: rgba(248,250,252,0.96);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(15,23,42,0.07);
}

/* ---------- Header ---------- */
.app-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.2rem; font-weight: 800; line-height: 1.15;
    color: #0f172a; letter-spacing: -0.4px; margin-bottom: 0.3rem;
}
.app-subtitle {
    font-size: 1.0rem; color: #64748b; line-height: 1.6; margin-bottom: 1.1rem;
}

/* ---------- Pills ---------- */
.pill {
    display: inline-block; padding: 0.22rem 0.65rem; border-radius: 5px;
    background: #f1f5f9; border: 1px solid #e2e8f0; color: #475569;
    font-size: 0.76rem; font-weight: 600; margin-right: 0.38rem; margin-bottom: 0.38rem;
    font-family: 'DM Mono', monospace; letter-spacing: 0.3px;
}
.pill-primary {background:#eff6ff; border-color:#bfdbfe; color:#1d4ed8;}

/* ---------- KPI metric cards ---------- */
.metric-card {
    padding: 1.15rem 1.2rem 1.05rem;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06), 0 4px 14px rgba(15,23,42,0.04);
    min-height: 128px;
    position: relative; overflow: hidden;
}
.metric-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 3px; background: var(--accent, #2563eb); border-radius: 12px 12px 0 0;
}
.metric-card-primary { --accent: #2563eb; }
.metric-card-orange  { --accent: #ea580c; }
.metric-card-green   { --accent: #16a34a; }
.metric-card-slate   { --accent: #475569; }

.metric-label {
    font-size: 0.75rem; color: #64748b; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 0.38rem;
}
.metric-value {
    font-size: 2.0rem; font-weight: 700; color: #0f172a; line-height: 1.1;
    font-family: 'DM Mono', monospace;
}
.metric-note { font-size: 0.79rem; color: #64748b; margin-top: 0.38rem; line-height: 1.4; }
.metric-delta-pos {color:#16a34a; font-size:0.80rem; font-weight:700;}
.metric-delta-neg {color:#dc2626; font-size:0.80rem; font-weight:700;}

/* ---------- Alert boxes ---------- */
.alert {
    padding: 0.88rem 1rem; border-radius: 10px; margin: 0.4rem 0 0.6rem;
    font-size: 0.91rem; line-height: 1.5;
    border-left: 4px solid transparent;
}
.alert-warn    {background:#fffbeb; border-color:#f59e0b; color:#78350f;}
.alert-success {background:#f0fdf4; border-color:#22c55e; color:#14532d;}
.alert-info    {background:#eff6ff; border-color:#3b82f6; color:#1e3a8a;}
.alert-danger  {background:#fef2f2; border-color:#ef4444; color:#7f1d1d;}

/* ---------- Recommend card ---------- */
.recommend-card {
    padding: 1.15rem 1.25rem; border-radius: 12px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    box-shadow: 0 1px 6px rgba(15,23,42,0.05); margin-bottom: 1rem;
}
.recommend-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; font-weight: 700; color: #0f172a; margin-bottom: 0.45rem;
}
.recommend-body { font-size: 0.91rem; color: #334155; line-height: 1.65; }
.mono {font-family:'DM Mono',monospace; font-weight:500;}

/* ---------- Section divider ---------- */
.section-rule {
    border: none; height: 1px;
    background: linear-gradient(90deg, #e2e8f0 60%, transparent);
    margin: 1.4rem 0;
}

/* ---------- Sensitivity mini-cards ---------- */
.sens-grid {display:flex; gap:0.55rem; flex-wrap:wrap; margin:0.8rem 0 1.1rem;}
.sens-card {
    flex: 1; min-width: 115px;
    padding: 0.75rem 0.85rem; border-radius: 10px;
    border: 1px solid #e2e8f0; background: #ffffff;
    box-shadow: 0 1px 4px rgba(15,23,42,0.04);
}
.sens-name {font-size:0.72rem; color:#64748b; font-weight:600; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.28rem;}
.sens-val  {font-size:1.15rem; font-weight:700; font-family:'DM Mono',monospace; line-height:1.1;}
.sens-ref  {font-size:0.70rem; color:#94a3b8; margin-top:0.18rem;}
.sens-pos  {color:#16a34a;}
.sens-neg  {color:#dc2626;}
.sens-neu  {color:#64748b;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background:#f8fafc; border-right:1px solid #e2e8f0; }
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family:'DM Sans',sans-serif; font-size:0.72rem; font-weight:700;
    color:#94a3b8; text-transform:uppercase; letter-spacing:1.1px;
    margin-top:1.1rem; margin-bottom:0.35rem;
}

/* ---------- Chart polish ---------- */
.stPlotlyChart { border-radius:10px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid #e2e8f0 !important; border-radius:10px !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session state
# ============================================================
def _init_state() -> None:
    for k, v in DEFAULT_INPUTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _apply_sample(name: str) -> None:
    for k, v in SAMPLE_MIXES[name].items():
        st.session_state[k] = v

_init_state()


# ============================================================
# Loading
# ============================================================
def _infer_type(info: Dict[str, Any]) -> str:
    t = str(info.get("type", "")).strip()
    if t and t != "-":
        return t
    txt = f"{info.get('name','')} {info.get('file','')}".lower()
    for kw, label in [
        ("random forest","Random Forest"),("rf","Random Forest"),
        ("xgboost","XGBoost"),("xgb","XGBoost"),
        ("lightgbm","LightGBM"),("lgb","LightGBM"),
        ("catboost","CatBoost"),("cb","CatBoost"),
        ("gradient","Gradient Boosting"),("gbr","Gradient Boosting"),
        ("decision","Decision Tree"),("dt","Decision Tree"),
        ("ann","ANN"),("mlp","ANN"),
        ("ridge","Ridge"),("lasso","Lasso"),
        ("linear","Linear Regression"),("svr","SVR"),
    ]:
        if kw in txt:
            return label
    return "Other"


@st.cache_data
def load_registry() -> List[Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)
    else:
        reg = [
            {"name":"Random Forest Tuned",   "file":"models/tuned_rf.pkl","type":"Random Forest",   "status":"Primary / Best","show_in_comparison":True},
            {"name":"Random Forest Default", "file":"models/rf.pkl",       "type":"Random Forest",   "status":"Comparison",    "show_in_comparison":True},
            {"name":"XGBoost Default",       "file":"models/xgb.pkl",      "type":"XGBoost",         "status":"Comparison",    "show_in_comparison":True},
            {"name":"LightGBM Default",      "file":"models/lgb.pkl",      "type":"LightGBM",        "status":"Comparison",    "show_in_comparison":True},
            {"name":"CatBoost Default",      "file":"models/cb.pkl",       "type":"CatBoost",        "status":"Comparison",    "show_in_comparison":True},
        ]
    out = []
    for item in reg:
        item = dict(item)
        item.setdefault("name", "Unnamed"); item.setdefault("file", "")
        # Strip variant suffixes from names loaded from registry
        import re as _re
        item["name"] = _re.sub(r'\s*[—\-]+\s*Scenario\s+B\s*$', '', item["name"], flags=_re.IGNORECASE).strip()
        item["type"] = _infer_type(item)
        item.setdefault("status", "Comparison"); item.setdefault("show_in_comparison", True)
        for m in ("r2", "mae", "rmse"):
            item.setdefault(m, None)
        if item["name"] == PRIMARY_MODEL_NAME:
            item["status"] = "Primary / Best"
        out.append(item)
    out.sort(key=lambda x: 0 if x.get("name") == PRIMARY_MODEL_NAME else 1)
    return out


@st.cache_data
def load_pareto_data() -> pd.DataFrame:
    if not PARETO_PATH.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS + ["Predicted_Strength", "Active_SCM"])
    df = pd.read_csv(PARETO_PATH)
    df.columns = [str(c).strip() for c in df.columns]
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "Predicted_Strength" not in df.columns:
        matched = next((c for c in STRENGTH_COLUMN_CANDIDATES if c in df.columns), None)
        df["Predicted_Strength"] = pd.to_numeric(df[matched], errors="coerce") if matched else np.nan
    else:
        df["Predicted_Strength"] = pd.to_numeric(df["Predicted_Strength"], errors="coerce")
    if "Active_SCM" not in df.columns:
        df["Active_SCM"] = df[SCM_COLUMNS].idxmax(axis=1)
    df["Active_SCM"] = df["Active_SCM"].astype(str)
    return df.dropna(subset=["Predicted_Strength"]).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_model(model_file: str) -> Tuple[Any, str]:
    path = APP_DIR / model_file
    if not path.exists():
        return None, f"File not found: {model_file}"
    last_err = ""
    loaders = []
    if joblib is not None:
        loaders.append(("joblib", lambda p: joblib.load(p)))
    loaders.append(("pickle", lambda p: pickle.load(open(p, "rb"))))
    for label, loader in loaders:
        try:
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                return loader(path), "Loaded"
        except Exception as e:
            last_err = f"{label}: {e}"
    return None, f"Could not load {model_file}. {last_err}"


# ============================================================
# Input helpers
# ============================================================
def build_input(water: float, cement: float, scm_type: str, scm_amount: float, fiber: float) -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row.update({"Water": float(water), "Cement": float(cement), "Fiber": float(fiber)})
    if scm_type in SCM_COLUMNS:
        row[scm_type] = float(scm_amount)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def row_to_ui(row: pd.DataFrame) -> Dict[str, Any]:
    active = max(SCM_COLUMNS, key=lambda c: float(row.loc[0, c]))
    return {"Water": float(row.loc[0, "Water"]), "Cement": float(row.loc[0, "Cement"]),
            "SCM_Type": active, "SCM_Amount": float(row.loc[0, active]), "Fiber": float(row.loc[0, "Fiber"])}


# ============================================================
# Prediction helpers
# ============================================================
def _valid(v: float) -> bool:
    try: return np.isfinite(float(v)) and 0.0 <= float(v) <= 20.0
    except Exception: return False


def predict(row: pd.DataFrame, info: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Returns (prediction, status). Returns (None, error_message) if the model
    cannot be loaded or prediction fails. No demo/fallback values are used.
    """
    model, status = load_model(info.get("file", ""))
    if model is None:
        return None, f"Model not available: {status}"
    x = row[FEATURE_COLUMNS].copy()
    for attempt in (x, x.to_numpy()):
        try:
            p = round(float(model.predict(attempt)[0]), 4)
            if _valid(p):
                return p, "Real model"
            return None, f"Prediction {p} is outside the valid tensile range (0–20 MPa)."
        except Exception:
            continue
    return None, "Prediction failed — check model file compatibility."

def rf_uncertainty(row: pd.DataFrame, info: Dict[str, Any]) -> Tuple[Optional[float], int]:
    model, _ = load_model(info.get("file", ""))
    if model is None: return None, 0
    ens = model
    for attr in ("named_steps", "steps"):
        if hasattr(model, attr):
            steps_iter = list(model.named_steps.values()) if attr == "named_steps" else [s for _, s in model.steps]
            for step in reversed(steps_iter):
                if hasattr(step, "estimators_"):
                    ens = step; break
    if not hasattr(ens, "estimators_"): return None, 0
    x = row[FEATURE_COLUMNS].copy(); x_np = x.to_numpy(); preds = []
    for est in ens.estimators_:
        for attempt in (x, x_np):
            try:
                p = float(np.ravel(est.predict(attempt))[0])
                if _valid(p): preds.append(p); break
            except Exception: continue
    return (float(np.std(preds)), len(preds)) if len(preds) >= 2 else (None, len(preds))


def check_ranges(ui: Dict[str, Any]) -> List[str]:
    out = []
    for k in ["Water","Cement","SCM_Amount","Fiber"]:
        v = float(ui[k]); lim = TRAINING_RANGES[k]
        if   v < lim["min"]: out.append(f"{k.replace('_',' ')} ({v:g} {lim['unit']}) is below training range ({lim['min']:g}–{lim['max']:g}).")
        elif v > lim["max"]: out.append(f"{k.replace('_',' ')} ({v:g} {lim['unit']}) exceeds training range ({lim['min']:g}–{lim['max']:g}).")
    return out


def reliability_score(ui: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    score = 100.0; notes = []
    for k in ["Water","Cement","SCM_Amount","Fiber"]:
        v = float(ui[k]); lim = TRAINING_RANGES[k]; mn, mx = float(lim["min"]), float(lim["max"]); w = mx - mn
        sc = (v - mn) / w if w else 0.5
        if v < mn or v > mx:
            dist = abs(v - mn) if v < mn else abs(v - mx)
            score -= 30.0 + min(35.0, 100.0 * dist / max(w, 1e-9))
            notes.append(f"{k.replace('_',' ')} is outside the training range.")
        elif sc <= 0.05 or sc >= 0.95: score -= 10.0; notes.append(f"{k.replace('_',' ')} is at a training-range boundary.")
        elif sc <= 0.15 or sc >= 0.85: score -= 5.0;  notes.append(f"{k.replace('_',' ')} is near a training-range boundary.")
    s = int(round(max(0.0, min(100.0, score))))
    label = "High" if s >= 85 else ("Medium" if s >= 60 else "Low")
    if not notes: notes.append("All inputs are comfortably inside the training domain.")
    return s, label, notes


def _fmt_metric(info: Dict[str, Any], k: str) -> Any:
    v = info.get(k)
    if v is None or v == "": return "—"
    try: return round(float(v), 4)
    except Exception: return v


def comparison_table(row: pd.DataFrame, registry: List[Dict[str, Any]], primary: Dict[str, Any]) -> pd.DataFrame:
    pri_pred, pri_mode = predict(row, primary)
    if pri_pred is None:
        pri_pred = float("nan")
    records = []
    for info in registry:
        if not info.get("show_in_comparison", True): continue
        if _infer_type(info) not in RELIABLE_MODEL_TYPES: continue
        p, mode = predict(row, info)
        if p is None or not _valid(p): continue
        records.append({
            "Model": info["name"], "Type": _infer_type(info),
            "Prediction (MPa)": round(float(p), 4),
            "Δ vs RF Tuned": round(float(p - pri_pred), 4),
            "R²": _fmt_metric(info, "r2"), "MAE": _fmt_metric(info, "mae"),
            "Status": info.get("status", ""),
            "_sort": 0 if info["name"] == PRIMARY_MODEL_NAME else 1,
        })
    if not records:
        records.append({
            "Model": primary["name"], "Type": _infer_type(primary),
            "Prediction (MPa)": round(float(pri_pred), 4), "Δ vs RF Tuned": 0.0,
            "R²": _fmt_metric(primary,"r2"), "MAE": _fmt_metric(primary,"mae"),
            "Status": "Primary / Best", "_sort": 0,
        })
    df = pd.DataFrame(records).sort_values(["_sort","Prediction (MPa)"], ascending=[True,False]).reset_index(drop=True)
    return df.drop(columns=["_sort"], errors="ignore")


# ============================================================
# Explanation helpers
# ============================================================
def sensitivity_contributions(row: pd.DataFrame, info: Dict[str, Any], pred: float) -> pd.DataFrame:
    ui = row_to_ui(row); records = []
    refs = [
        ("Water",      TRAINING_RANGES["Water"]["reference"],      "Effect vs reference water content."),
        ("Cement",     TRAINING_RANGES["Cement"]["reference"],     "Effect vs reference cement content."),
        ("SCM_Amount", TRAINING_RANGES["SCM_Amount"]["reference"], "Effect vs reference SCM dosage."),
        ("Fiber",      TRAINING_RANGES["Fiber"]["reference"],      "Effect vs reference fiber content."),
    ]
    for feat, ref, note in refs:
        alt = {**ui, feat: ref}
        alt_row = build_input(alt["Water"], alt["Cement"], alt["SCM_Type"], alt["SCM_Amount"], alt["Fiber"])
        rp, _ = predict(alt_row, info)
        contrib = round(float(pred - rp), 4) if rp is not None else 0.0
        records.append({"Feature": feat.replace("_"," "), "Current Value": ui[feat], "Reference": ref,
                         "Contribution (MPa)": contrib, "Interpretation": note})
    if ui["SCM_Type"] != "Quartz":
        qr = build_input(ui["Water"], ui["Cement"], "Quartz", ui["SCM_Amount"], ui["Fiber"])
        qp, _ = predict(qr, info)
        records.append({"Feature":"SCM Type","Current Value":ui["SCM_Type"],"Reference":"Quartz",
                         "Contribution (MPa)":round(float(pred-qp),4) if qp is not None else 0.0,
                         "Interpretation":"SCM material effect vs Quartz."})
    else:
        zr = build_input(ui["Water"], ui["Cement"], "Quartz", 0.0, ui["Fiber"])
        zp, _ = predict(zr, info)
        records.append({"Feature":"SCM Type","Current Value":"Quartz","Reference":"Quartz = 0",
                         "Contribution (MPa)":round(float(pred-zp),4) if zp is not None else 0.0,
                         "Interpretation":"Quartz contribution vs no SCM."})
    df = pd.DataFrame(records)
    return df.reindex(df["Contribution (MPa)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _explanation_text(contrib_df: pd.DataFrame, pred: float) -> str:
    pos = contrib_df[contrib_df["Contribution (MPa)"] > 0].head(2)
    neg = contrib_df[contrib_df["Contribution (MPa)"] < 0].head(2)
    parts = [f"The selected model predicts **{pred:.3f} MPa** tensile strength for the current mix design."]
    if not pos.empty:
        parts.append("Strongest positive drivers: " + ", ".join(f"**{r['Feature']}** ({r['Contribution (MPa)']:+.3f} MPa)" for _, r in pos.iterrows()) + ".")
    if not neg.empty:
        parts.append("Strongest reducing factors: " + ", ".join(f"**{r['Feature']}** ({r['Contribution (MPa)']:+.3f} MPa)" for _, r in neg.iterrows()) + ".")
    parts.append("These are sensitivity estimates, not formal SHAP values. Validate experimentally before deployment.")
    return " ".join(parts)


# ============================================================
# Chart factories
# ============================================================
_FONT = dict(family="DM Sans, sans-serif")

def _layout(**kw) -> dict:
    """Base layout dict. Does NOT include font= to avoid duplicate-keyword errors
    when callers also set title=dict(font=...). Add font=_FONT at each call site."""
    base = dict(template=_PLOTLY_TEMPLATE,
                plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                margin=dict(l=10, r=20, t=55, b=35))
    base.update(kw)
    return base


def _chart_contribution(contrib_df: pd.DataFrame) -> go.Figure:
    df = contrib_df.sort_values("Contribution (MPa)", ascending=True).copy()
    max_abs = max(0.05, float(df["Contribution (MPa)"].abs().max()))
    colors = ["#16a34a" if v >= 0 else "#dc2626" for v in df["Contribution (MPa)"]]
    fig = go.Figure(go.Bar(
        x=df["Contribution (MPa)"], y=df["Feature"], orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(15,23,42,0.1)", width=1)),
        text=[f"{v:+.3f}" for v in df["Contribution (MPa)"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:+.4f} MPa<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1.5, line_dash="dot", line_color="#94a3b8")
    fig.update_layout(
        title=dict(text="Local Sensitivity — Feature Contributions", font=dict(size=14)),
        xaxis=dict(title="ΔStrength (MPa)", range=[-1.4*max_abs, 1.4*max_abs],
                   zeroline=False, gridcolor="#f1f5f9"),
        yaxis=dict(title="", gridcolor="#f1f5f9"),
        height=400, font=_FONT, **_layout(margin=dict(l=10, r=90, t=55, b=40)),
    )
    return fig


def _chart_radar(ui: Dict[str, Any]) -> go.Figure:
    labels = ["Water","Cement","SCM Amount","Fiber"]
    vals = []
    for k in ["Water","Cement","SCM_Amount","Fiber"]:
        lim = TRAINING_RANGES[k]; sc = (float(ui[k]) - lim["min"]) / (lim["max"] - lim["min"])
        vals.append(max(0.0, min(1.0, sc)))
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=labels+[labels[0]],
        fill="toself", name="Mix profile",
        line=dict(color="#2563eb", width=2.5), fillcolor="rgba(37,99,235,0.10)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5]*5, theta=labels+[labels[0]], mode="lines",
        line=dict(color="#cbd5e1", width=1.5, dash="dash"),
        name="Reference midpoint", showlegend=False,
    ))
    fig.update_layout(
        title=dict(text="Mix Profile vs Training Range", font=dict(size=14)),
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], tickvals=[0,0.5,1],
                            ticktext=["Min","Mid","Max"], gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0"), bgcolor="#ffffff",
        ),
        showlegend=False, height=370,
        font=_FONT, **_layout(margin=dict(l=30, r=30, t=60, b=30)),
    )
    return fig


def _chart_gauge(score: int, label: str) -> go.Figure:
    color = "#16a34a" if score >= 85 else ("#f59e0b" if score >= 60 else "#ef4444")
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        title={"text":f"Input Reliability<br><span style='font-size:0.82em;color:#64748b;font-family:DM Sans'>{label}</span>",
               "font":{"family":"DM Sans","size":14}},
        number={"suffix":"%","font":{"family":"DM Mono","size":38}},
        gauge={
            "axis": {"range":[0,100],"tickcolor":"#94a3b8"},
            "bar":  {"color":color,"thickness":0.28},
            "bgcolor":"#f8fafc","borderwidth":0,
            "steps":[{"range":[0,60],"color":"#fee2e2"},{"range":[60,85],"color":"#fef9c3"},{"range":[85,100],"color":"#dcfce7"}],
        },
    ))
    fig.update_layout(height=260, font=_FONT, **_layout(margin=dict(l=20,r=20,t=55,b=10)))
    return fig


def _chart_waterfall(contrib_df: pd.DataFrame, pred: float) -> go.Figure:
    df = contrib_df.sort_values("Contribution (MPa)", key=lambda s: s.abs(), ascending=False).copy()
    total = float(df["Contribution (MPa)"].sum()); base = float(pred - total)
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute"]+["relative"]*len(df)+["total"],
        x=["Baseline"]+df["Feature"].tolist()+["Prediction"],
        y=[base]+df["Contribution (MPa)"].tolist()+[pred],
        text=[f"{base:.3f}"]+[f"{v:+.3f}" for v in df["Contribution (MPa)"]]+[f"{pred:.3f}"],
        textposition="outside", cliponaxis=False,
        connector={"line":{"color":"#cbd5e1","width":1.5,"dash":"dot"}},
        increasing={"marker":{"color":"#16a34a"}},
        decreasing={"marker":{"color":"#dc2626"}},
        totals={"marker":{"color":"#2563eb"}},
    ))
    fig.update_layout(
        title=dict(text="Waterfall: How Each Feature Shapes the Prediction", font=dict(size=14)),
        yaxis=dict(title="Predicted Strength (MPa)", gridcolor="#f1f5f9"),
        xaxis=dict(tickangle=-18), showlegend=False,
        height=420, font=_FONT, **_layout(margin=dict(l=15,r=20,t=60,b=90)),
    )
    return fig


def _chart_whatif(row: pd.DataFrame, info: Dict[str, Any], feature: str) -> go.Figure:
    ui = row_to_ui(row); lim = TRAINING_RANGES[feature]
    vals = np.linspace(lim["min"], lim["max"], 55); rows = []
    for v in vals:
        alt = {**ui, feature: float(v)}
        ar = build_input(alt["Water"], alt["Cement"], alt["SCM_Type"], alt["SCM_Amount"], alt["Fiber"])
        p, _ = predict(ar, info)
        if p is not None:
            rows.append({feature.replace("_"," "): float(v), "Strength (MPa)": p})
    df = pd.DataFrame(rows); x_col = feature.replace("_"," ")
    fig = px.line(df, x=x_col, y="Strength (MPa)", template=_PLOTLY_TEMPLATE,
                  title=f"Sensitivity: Effect of {x_col} on Predicted Strength")
    fig.update_traces(line=dict(color="#2563eb", width=2.5), mode="lines")
    cur = ui[feature]
    fig.add_vline(x=cur, line_dash="dash", line_color="#f59e0b", line_width=2,
                  annotation_text=f"Current: {cur:g}", annotation_font=dict(color="#f59e0b", size=11))
    fig.add_vrect(x0=lim["min"], x1=lim["max"], fillcolor="#dbeafe", opacity=0.12, layer="below", line_width=0)
    fig.update_layout(height=400, **_layout(),
                      font=_FONT,
                      yaxis=dict(gridcolor="#f1f5f9", title="Predicted Strength (MPa)"),
                      xaxis=dict(gridcolor="#f1f5f9"))
    return fig


def _chart_scm_choice(row: pd.DataFrame, info: Dict[str, Any]) -> go.Figure:
    ui = row_to_ui(row); rows = []
    for scm in SCM_COLUMNS:
        ar = build_input(ui["Water"], ui["Cement"], scm, ui["SCM_Amount"], ui["Fiber"])
        p, _ = predict(ar, info)
        if p is not None:
            rows.append({"SCM Type": scm, "Predicted Strength (MPa)": p})
    df = pd.DataFrame(rows).sort_values("Predicted Strength (MPa)", ascending=False)
    colors = ["#2563eb" if scm == ui["SCM_Type"] else "#94a3b8" for scm in df["SCM Type"]]
    fig = go.Figure(go.Bar(
        x=df["SCM Type"], y=df["Predicted Strength (MPa)"],
        text=[f"{v:.3f}" for v in df["Predicted Strength (MPa)"]], textposition="outside",
        marker=dict(color=colors, line=dict(color="rgba(15,23,42,0.08)", width=1)),
        hovertemplate="<b>%{x}</b><br>%{y:.4f} MPa<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="SCM Comparison at Same Dosage — Current SCM in Blue", font=dict(size=14)),
        yaxis=dict(title="Predicted Strength (MPa)", gridcolor="#f1f5f9"),
        xaxis=dict(title=""), showlegend=False,
        font=_FONT, height=370, **_layout(margin=dict(l=10,r=20,t=55,b=55)),
    )
    return fig


def _chart_comparison_bar(df: pd.DataFrame) -> go.Figure:
    plot = df[df["Prediction (MPa)"].apply(_valid)].sort_values("Prediction (MPa)", ascending=True).copy()
    colors = ["#2563eb" if s == "Primary / Best" else "#94a3b8" for s in plot["Status"]]
    fig = go.Figure(go.Bar(
        x=plot["Prediction (MPa)"], y=plot["Model"], orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(15,23,42,0.08)", width=1)),
        text=[f"{v:.3f}" for v in plot["Prediction (MPa)"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>%{x:.4f} MPa<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Model Ensemble Comparison — Primary in Blue", font=dict(size=14)),
        xaxis=dict(title="Predicted Tensile Strength (MPa)", gridcolor="#f1f5f9"),
        yaxis=dict(title=""), showlegend=False,
        font=_FONT, height=430, **_layout(margin=dict(l=10,r=80,t=55,b=35)),
    )
    return fig


def _chart_agreement(compare_df: pd.DataFrame, pred: float) -> go.Figure:
    valid = compare_df[compare_df["Prediction (MPa)"].apply(_valid)]
    mean_p = float(valid["Prediction (MPa)"].mean())
    lo = float(valid["Prediction (MPa)"].min()); hi = float(valid["Prediction (MPa)"].max())
    fig = go.Figure(go.Indicator(
        mode="number+delta", value=pred,
        delta={"reference":mean_p,"relative":False,"valueformat":".3f",
               "increasing":{"color":"#16a34a"},"decreasing":{"color":"#dc2626"}},
        title={"text":f"Selected vs Ensemble Mean<br>"
                      f"<span style='font-size:0.78em;color:#64748b;font-family:DM Sans'>Band: {lo:.3f} – {hi:.3f} MPa</span>",
               "font":{"family":"DM Sans","size":14}},
        number={"suffix":" MPa","font":{"family":"DM Mono","size":38}},
    ))
    fig.update_layout(height=240, font=_FONT, **_layout(margin=dict(l=20,r=20,t=55,b=10)))
    return fig


# ============================================================
# Optimizer helpers
# ============================================================
def _score_pareto(df: pd.DataFrame, goal: str) -> pd.DataFrame:
    out = df.copy()
    def nrm(s):
        return pd.Series(np.ones(len(s)), index=s.index) if s.max() == s.min() else (s - s.min()) / (s.max() - s.min())
    if goal == "Maximum strength":
        out["Score"] = out["Predicted_Strength"]
    elif goal == "Minimum cement":
        out["Score"] = -out["Cement"]
        return out.sort_values(["Score","Predicted_Strength"], ascending=[False,False])
    elif goal == "Minimum fiber":
        out["Score"] = -out["Fiber"]
        return out.sort_values(["Score","Predicted_Strength"], ascending=[False,False])
    else:
        out["Score"] = 0.55*nrm(out["Predicted_Strength"]) - 0.22*nrm(out["Cement"]) - 0.13*nrm(out["Water"]) - 0.10*nrm(out["Fiber"])
    return out.sort_values("Score", ascending=False)


def filter_pareto(pareto: pd.DataFrame, scm: str, max_c: float, max_w: float, max_f: float, min_s: float, goal: str) -> pd.DataFrame:
    if pareto.empty: return pareto
    df = pareto.copy()
    if scm != "Any SCM":
        df = df[df["Active_SCM"].astype(str) == scm]
    df = df[(df["Cement"] <= max_c) & (df["Water"] <= max_w) & (df["Fiber"] <= max_f) & (df["Predicted_Strength"] >= min_s)]
    return _score_pareto(df, goal).reset_index(drop=True) if not df.empty else df


# ============================================================
# Report
# ============================================================
def generate_report(ui, model_name, pred, mode, unc_text, rel_score, rel_label, warns, cmp_df) -> str:
    lines = [
        "SCENARIO B — TENSILE STRENGTH PREDICTION REPORT",
        "="*54,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}","",
        "INPUT MIX DESIGN","-"*54,
        f"  Water:      {ui['Water']:.2f} kg/m³",
        f"  Cement:     {ui['Cement']:.2f} kg/m³",
        f"  SCM Type:   {ui['SCM_Type']}",
        f"  SCM Amount: {ui['SCM_Amount']:.2f} kg/m³",
        f"  Fiber:      {ui['Fiber']:.3f} %","",
        "PREDICTION OUTPUT","-"*54,
        f"  Model:      {model_name}",
        f"  Mode:       {mode}",
        f"  Prediction: {pred:.4f} MPa",
        f"  Uncertainty:{unc_text}",
        f"  Reliability:{rel_score}% ({rel_label})","",
        "TRAINING RANGE STATUS","-"*54,
    ]
    lines += [f"  WARNING: {w}" for w in warns] or ["  All inputs are within the training range."]
    lines += ["","MODEL COMPARISON","-"*54]
    if not cmp_df.empty:
        cols = [c for c in ["Model","Prediction (MPa)","Δ vs RF Tuned","Status"] if c in cmp_df.columns]
        lines.append(cmp_df[cols].to_string(index=False))
    lines += ["","NOTE: Validate experimentally before practical mix-design adoption."]
    return "\n".join(lines)


# ============================================================
# Load shared resources
# ============================================================
registry    = load_registry()
pareto_df   = load_pareto_data()
primary     = next((m for m in registry if m.get("name") == PRIMARY_MODEL_NAME), registry[0])
model_names = [m["name"] for m in registry]


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Model")
    sel_name = st.selectbox(
        "Active model", model_names,
        index=model_names.index(PRIMARY_MODEL_NAME) if PRIMARY_MODEL_NAME in model_names else 0,
        help="RF Tuned is the recommended primary model.",
    )
    sel_info = next(m for m in registry if m["name"] == sel_name)

    st.markdown("### Recommended")
    st.caption("⭐ Random Forest Tuned")
    st.caption("Use other models for comparison only.")

    st.markdown("### Training Ranges")
    for k, v in TRAINING_RANGES.items():
        st.caption(f"**{k.replace('_',' ')}**: {v['min']:g} – {v['max']:g} {v['unit']}")

    st.markdown("### Model Status")
    ok = sum(1 for m in registry if load_model(m.get("file",""))[0] is not None)
    for m in registry:
        loaded = load_model(m.get("file",""))[0] is not None
        st.caption(("🟢 " if loaded else "🔴 ") + m["name"].split("—")[0].strip())
    st.caption(f"{ok}/{len(registry)} models loaded.")

    st.markdown("---")
    if st.button("🔄 Clear cache", use_container_width=True):
        st.cache_data.clear(); st.cache_resource.clear(); st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================
st.markdown('<div class="app-title">TensAile Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Machine Learning prediction dashboard to predict tensile strength</div>',
    unsafe_allow_html=True,
)
# Section switcher (no pills)
section = st.selectbox(
    "Section",
    ["Predictor", "Explanation", "Optimizer"],
    index=0,
    label_visibility="collapsed",
    help="Switch between Predictor, Explanation and Optimizer views.",
)
st.markdown(
    '<div class="alert alert-info" style="margin-top:0.7rem;">'
    'ℹ️ This page presents <b>114 mix designs</b> identified by <b>NSGA-II</b> '
    '(Non-dominated Sorting Genetic Algorithm II) as Pareto-optimal — meaning no other '
    'mix design simultaneously delivers <b>higher strength</b> AND <b>lower cement content</b>.'
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# INPUT PANEL
# ============================================================
with st.expander("🔧 Input Mix Design", expanded=True):
    st.markdown("**Quick demo mixes**")
    sb1, sb2, sb3, sb4 = st.columns([1,1,1,2.2])
    with sb1:
        if st.button("Balanced",     use_container_width=True): _apply_sample("Balanced Mix")
    with sb2:
        if st.button("Low Cement",   use_container_width=True): _apply_sample("Low Cement Mix")
    with sb3:
        if st.button("High Strength",use_container_width=True): _apply_sample("High Strength Trial")
    with sb4:
        st.caption("Preset examples for benchmarking or demos.")

    c1, c2, c3, c4, c5 = st.columns([1,1,1.25,1,1])
    with c1:
        water      = st.number_input("Water (kg/m³)",     min_value=250.0, max_value=450.0,  step=1.0,    key="Water")
    with c2:
        cement     = st.number_input("Cement (kg/m³)",    min_value=600.0, max_value=1200.0, step=1.0,    key="Cement")
    with c3:
        if st.session_state.get("SCM_Type") not in SCM_TYPES:
            st.session_state["SCM_Type"] = SCM_TYPES[0]
        scm_type   = st.selectbox("SCM Type", SCM_TYPES, key="SCM_Type")
    with c4:
        scm_amount = st.number_input("SCM Amount (kg/m³)", min_value=0.0,  max_value=350.0,  step=1.0,    key="SCM_Amount")
    with c5:
        fiber      = st.number_input("Fiber (%)",          min_value=0.0,  max_value=0.50,   step=0.01, format="%.3f", key="Fiber")


# ============================================================
# SHARED COMPUTATIONS
# ============================================================
row     = build_input(water, cement, scm_type, scm_amount, fiber)
ui_vals = row_to_ui(row)

warns   = check_ranges(ui_vals)
rel_s, rel_label, rel_notes = reliability_score(ui_vals)

sel_pred, sel_mode = predict(row, sel_info)
pri_pred, pri_mode = predict(row, primary)

# sel_pred / pri_pred are None when the model file is missing or prediction fails.
# All UI sections guard against None before displaying.

cmp_df    = comparison_table(row, registry, primary)
valid_cmp = cmp_df[cmp_df["Prediction (MPa)"].apply(_valid)].copy()

spread   = float(valid_cmp["Prediction (MPa)"].std()) if len(valid_cmp) > 1 else 0.0
min_pred = float(valid_cmp["Prediction (MPa)"].min()) if len(valid_cmp) > 0 else (sel_pred or 0.0)
max_pred = float(valid_cmp["Prediction (MPa)"].max()) if len(valid_cmp) > 0 else (sel_pred or 0.0)

rf_std, rf_n = rf_uncertainty(row, primary)
unc_val  = rf_std if rf_std is not None else spread
unc_note = f"RF tree uncertainty ({rf_n} trees)" if rf_std is not None else "Model agreement spread"
unc_text = f"±{unc_val:.3f} MPa ({unc_note})"

if sel_name != PRIMARY_MODEL_NAME:
    st.markdown(
        f'<div class="alert alert-warn">⚠️ You selected <b>{sel_name}</b>. '
        f'Recommended model for final reporting: <b>{PRIMARY_MODEL_NAME}</b>.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTOR
# ============================================================
if section == "Predictor":

    # --- 4 KPI cards ---
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        if sel_pred is not None:
            delta = (sel_pred - pri_pred) if pri_pred is not None else 0.0
            d_html = (f'<span class="metric-delta-pos">▲ +{delta:.3f} vs RF Tuned</span>' if delta > 0.001
                      else f'<span class="metric-delta-neg">▼ {delta:.3f} vs RF Tuned</span>' if delta < -0.001
                      else '')
            val_html = f'{sel_pred:.3f} <span style="font-size:0.95rem;color:#64748b;font-family:DM Sans">MPa</span>'
        else:
            d_html = ''
            val_html = '<span style="font-size:1.1rem;color:#dc2626">Unavailable</span>'
        st.markdown(
            f'<div class="metric-card metric-card-primary">'
            f'<div class="metric-label">Predicted Tensile Strength</div>'
            f'<div class="metric-value">{val_html}</div>'
            f'{d_html}'
            f'<div class="metric-note">{sel_name.split("—")[0].strip()}<br>{sel_mode}</div>'
            f'</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(
            f'<div class="metric-card metric-card-orange">'
            f'<div class="metric-label">Uncertainty / Agreement</div>'
            f'<div class="metric-value">±{unc_val:.3f} <span style="font-size:0.95rem;color:#64748b;font-family:DM Sans">MPa</span></div>'
            f'<div class="metric-note">{unc_note}</div>'
            f'</div>', unsafe_allow_html=True)

    with col_c:
        rc = "#15803d" if rel_s >= 85 else ("#b45309" if rel_s >= 60 else "#b91c1c")
        st.markdown(
            f'<div class="metric-card metric-card-green">'
            f'<div class="metric-label">Input Reliability</div>'
            f'<div class="metric-value" style="color:{rc}">{rel_s}%</div>'
            f'<div class="metric-note">Level: <b>{rel_label}</b></div>'
            f'</div>', unsafe_allow_html=True)

    with col_d:
        sv = "✓ In Range" if not warns else f"⚠ {len(warns)} Warning{'s' if len(warns)>1 else ''}"
        sc_ = "#15803d" if not warns else "#b45309"
        st.markdown(
            f'<div class="metric-card metric-card-slate">'
            f'<div class="metric-label">Training Domain Status</div>'
            f'<div class="metric-value" style="font-size:1.45rem;color:{sc_}">{sv}</div>'
            f'<div class="metric-note">Valid-range check active</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

    left, right = st.columns([1.1, 1.35], gap="large")

    with left:
        st.markdown("#### Primary Model Result")
        pri_str  = f'<span class="mono">{pri_pred:.3f} MPa</span>' if pri_pred is not None else '<span style="color:#dc2626">Not available — check model file</span>'
        sel_str  = f'<span class="mono">{sel_pred:.3f} MPa</span>' if sel_pred is not None else '<span style="color:#dc2626">Not available — check model file</span>'
        st.markdown(
            f'<div class="recommend-card">'
            f'<div class="recommend-title">⭐ {PRIMARY_MODEL_NAME}</div>'
            f'<div class="recommend-body">'
            f'Primary RF prediction: {pri_str}<br>'
            f'Selected model ({sel_name.split("—")[0].strip()}): {sel_str}<br>'
            f'Ensemble spread (σ): <span class="mono">±{spread:.3f} MPa</span><br>'
            f'Reliability: <b>{rel_s}%</b> — {rel_label}'
            f'</div></div>', unsafe_allow_html=True)

        if warns:
            for w in warns:
                st.markdown(f'<div class="alert alert-warn">⚠️ {w}<br><span style="font-size:0.85em">Treat as extrapolation.</span></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert alert-success">✅ All inputs are within the training range.</div>', unsafe_allow_html=True)

        st.markdown("**Reliability notes**")
        for n in rel_notes:
            st.caption(f"• {n}")

        with st.expander("📋 Input vector"):
            st.dataframe(row, use_container_width=True, hide_index=True)

        report_txt = generate_report(
            ui_vals, sel_name,
            sel_pred if sel_pred is not None else float("nan"),
            sel_mode, unc_text, rel_s, rel_label, warns, cmp_df,
        )
        st.download_button(
            "⬇️ Download prediction report",
            data=report_txt.encode("utf-8"),
            file_name="prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with right:
        st.markdown("#### Model Comparison")
        show_cols = [c for c in ["Model","Type","Prediction (MPa)","Δ vs RF Tuned","R²","MAE","Status"] if c in cmp_df.columns]
        st.dataframe(cmp_df[show_cols], use_container_width=True, hide_index=True)
        if not valid_cmp.empty:
            st.plotly_chart(_chart_comparison_bar(valid_cmp), use_container_width=True)
        else:
            st.info("No reliable comparison models available for this input.")


# ============================================================
# EXPLANATION
# ============================================================
elif section == "Explanation":
    st.markdown("#### Explanation & Visual Diagnosis")
    st.markdown(
        '<div class="alert alert-info">Practical local explanation: each feature is set to its '
        'training-range reference while others stay fixed. The change in prediction shows each '
        'feature\'s directional contribution. Not formal SHAP — but fully transparent.</div>',
        unsafe_allow_html=True,
    )

    if sel_pred is None:
        st.markdown(
            '<div class="alert alert-danger">❌ The selected model could not produce a prediction. '            f'Reason: {sel_mode}. Please check that the model file exists and is compatible.</div>',
            unsafe_allow_html=True,
        )
        st.stop()

    contrib_df = sensitivity_contributions(row, sel_info, sel_pred)
    st.markdown(_explanation_text(contrib_df, sel_pred))

    # Sensitivity mini-cards
    st.markdown('<div class="sens-grid">', unsafe_allow_html=True)
    for _, r in contrib_df.iterrows():
        val = float(r["Contribution (MPa)"])
        css  = "sens-pos" if val > 0.005 else ("sens-neg" if val < -0.005 else "sens-neu")
        arrow = "↑" if val > 0.005 else ("↓" if val < -0.005 else "→")
        st.markdown(
            f'<div class="sens-card">'
            f'<div class="sens-name">{r["Feature"]}</div>'
            f'<div class="sens-val {css}">{arrow} {val:+.3f}</div>'
            f'<div class="sens-ref">Ref: {r["Reference"]}</div>'
            f'</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 1: contribution bar + radar
    r1a, r1b = st.columns([1.2, 1], gap="large")
    with r1a:
        st.plotly_chart(_chart_contribution(contrib_df), use_container_width=True)
    with r1b:
        st.plotly_chart(_chart_radar(ui_vals), use_container_width=True)

    # Row 2: waterfall + gauge + agreement indicator
    r2a, r2b = st.columns([1.2, 1], gap="large")
    with r2a:
        st.plotly_chart(_chart_waterfall(contrib_df, sel_pred), use_container_width=True)
    with r2b:
        st.plotly_chart(_chart_gauge(rel_s, rel_label), use_container_width=True)
        if len(valid_cmp) >= 2:
            st.plotly_chart(_chart_agreement(valid_cmp, sel_pred), use_container_width=True)

    # Contribution table
    st.markdown("#### Contribution Table")
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    # What-if curve
    st.markdown("#### What-if Sensitivity Curve")
    wc1, wc2 = st.columns([1, 3])
    with wc1:
        curve_feat = st.selectbox("Feature to vary", ["Water","Cement","SCM_Amount","Fiber"], index=3)
    with wc2:
        st.plotly_chart(_chart_whatif(row, sel_info, curve_feat), use_container_width=True)

    # SCM comparison
    st.markdown("#### SCM Type Comparison (same dosage)")
    st.plotly_chart(_chart_scm_choice(row, sel_info), use_container_width=True)


# ============================================================
# OPTIMIZER
# ============================================================
elif section == "Optimizer":
    st.markdown("#### Pareto-Based Optimal Mix Recommender")
    st.markdown(
        '<div class="alert alert-info">Filters <b>data/combined_pareto_front.csv</b> and ranks '
        'feasible mixes by your chosen design goal. Decision-support only — validate '
        'experimentally before adoption.</div>',
        unsafe_allow_html=True,
    )

    if pareto_df.empty:
        st.markdown('<div class="alert alert-danger">❌ Pareto file not found or has no valid strength column. '
                    'Ensure data/combined_pareto_front.csv is present.</div>', unsafe_allow_html=True)
        st.stop()

    oc1, oc2, oc3, oc4, oc5 = st.columns([1,1,1,1,1.15])
    with oc1: opt_scm    = st.selectbox("SCM filter", ["Any SCM"]+SCM_TYPES)
    with oc2: max_cement = st.number_input("Max cement (kg/m³)", min_value=750.0, max_value=1200.0, value=900.0, step=10.0)
    with oc3: max_water  = st.number_input("Max water (kg/m³)",  min_value=300.0, max_value=450.0,  value=380.0, step=5.0)
    with oc4: max_fiber  = st.number_input("Max fiber (%)",      min_value=0.0,   max_value=0.50,   value=0.30,  step=0.01, format="%.3f")
    with oc5: min_str    = st.number_input("Min strength (MPa)", min_value=0.0,   max_value=20.0,   value=3.0,   step=0.05)

    goal = st.selectbox("Optimization goal", ["Maximum strength","Minimum cement","Minimum fiber","Balanced mix"], index=3)

    results = filter_pareto(pareto_df, opt_scm, max_cement, max_water, max_fiber, min_str, goal)

    if results.empty:
        st.markdown('<div class="alert alert-warn">⚠️ No feasible mix found. Relax the constraints above.</div>', unsafe_allow_html=True)
    else:
        best = results.iloc[0]

        st.markdown(
            f'<div class="recommend-card">'
            f'<div class="recommend-title">🏆 Recommended Mix Design — {goal}</div>'
            f'<div class="recommend-body">'
            f'<b>Water:</b> {best["Water"]:.1f} kg/m³ &nbsp;|&nbsp; '
            f'<b>Cement:</b> {best["Cement"]:.1f} kg/m³ &nbsp;|&nbsp; '
            f'<b>SCM:</b> {best["Active_SCM"]} &nbsp;|&nbsp; '
            f'<b>Fiber:</b> {best["Fiber"]:.3f}%<br>'
            f'<b>Predicted tensile strength:</b> <span class="mono">{best["Predicted_Strength"]:.3f} MPa</span>'
            f'</div></div>', unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Water",    f"{float(best['Water']):.1f} kg/m³")
        m2.metric("Cement",   f"{float(best['Cement']):.1f} kg/m³")
        m3.metric("SCM",      str(best["Active_SCM"]))
        m4.metric("Fiber",    f"{float(best['Fiber']):.3f} %")
        m5.metric("Strength", f"{float(best['Predicted_Strength']):.3f} MPa")

        display_cols = [c for c in ["Water","Cement"]+SCM_COLUMNS+["Fiber","Predicted_Strength","Active_SCM","Score"] if c in results.columns]

        st.markdown("#### Top 25 Feasible Mixes")
        st.dataframe(results[display_cols].head(25), use_container_width=True, hide_index=True)

        p1, p2 = st.columns(2, gap="large")
        with p1:
            fig1 = px.scatter(
                results.head(300), x="Cement", y="Predicted_Strength",
                color="Active_SCM", size="Fiber",
                hover_data=["Water"]+SCM_COLUMNS,
                title="Cement vs Predicted Strength",
                template=_PLOTLY_TEMPLATE,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig1.add_scatter(
                x=[float(best["Cement"])], y=[float(best["Predicted_Strength"])],
                mode="markers+text", text=["★ Best"], textposition="top center",
                marker=dict(size=16, color="#0f172a", symbol="star"), name="Recommended",
            )
            fig1.update_layout(height=430, font=_FONT, **_layout())
            st.plotly_chart(fig1, use_container_width=True)

        with p2:
            fig2 = px.scatter(
                results.head(300), x="Water", y="Predicted_Strength",
                color="Active_SCM", size="Cement",
                hover_data=["Fiber"]+SCM_COLUMNS,
                title="Water vs Predicted Strength",
                template=_PLOTLY_TEMPLATE,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig2.add_scatter(
                x=[float(best["Water"])], y=[float(best["Predicted_Strength"])],
                mode="markers+text", text=["★ Best"], textposition="top center",
                marker=dict(size=16, color="#0f172a", symbol="star"), name="Recommended",
            )
            fig2.update_layout(height=430, font=_FONT, **_layout())
            st.plotly_chart(fig2, use_container_width=True)

        # Faceted scatter
        fig3 = px.scatter(
            results.head(500), x="Cement", y="Predicted_Strength",
            color="Active_SCM", size="Fiber",
            facet_col="Active_SCM" if results["Active_SCM"].nunique() <= 5 else None,
            title="Pareto Feasible Region by SCM Type",
            template=_PLOTLY_TEMPLATE,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig3.update_layout(height=440, font=_FONT, **_layout(margin=dict(l=10,r=10,t=60,b=35)))
        st.plotly_chart(fig3, use_container_width=True)

        # Parallel coordinates
        pc_cols = ["Water","Cement","Fiber","Predicted_Strength"]
        fig_pc = px.parallel_coordinates(
            results[pc_cols].head(150), color="Predicted_Strength",
            color_continuous_scale=px.colors.sequential.Blues,
            labels={"Predicted_Strength":"Strength (MPa)"},
            title="Parallel Coordinates — Top 150 Feasible Mixes",
            template=_PLOTLY_TEMPLATE,
        )
        fig_pc.update_layout(height=380, font=_FONT, **_layout(margin=dict(l=20,r=20,t=60,b=35)))
        st.plotly_chart(fig_pc, use_container_width=True)

        csv_bytes = results[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download optimized mixes CSV",
            data=csv_bytes, file_name="optimized_mixes.csv",
            mime="text/csv", use_container_width=True,
        )
