from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import joblib
except Exception:
    joblib = None


# ============================================================
# Scenario B Tensile Strength Predictor
# Main model: Random Forest Tuned — Scenario B
# Scenario B feature format:
# Water, Cement, Quartz, Fly Ash, Bagasse, Silica Fume,
# Calcium Carbonate, Fiber
# ============================================================

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
REGISTRY_PATH = APP_DIR / "model_registry.json"
PARETO_PATH = DATA_DIR / "combined_pareto_front.csv"

SCM_COLUMNS = ["Quartz", "Fly Ash", "Bagasse", "Silica Fume", "Calcium Carbonate"]
FEATURE_COLUMNS = ["Water", "Cement"] + SCM_COLUMNS + ["Fiber"]
SCM_TYPES = SCM_COLUMNS.copy()

PRIMARY_MODEL_NAME = "Random Forest Tuned — Scenario B"

TRAINING_RANGES = {
    "Water": {"min": 300.0, "max": 400.0, "unit": "kg/m³", "reference": 350.0},
    "Cement": {"min": 750.0, "max": 1000.0, "unit": "kg/m³", "reference": 875.0},
    "SCM_Amount": {"min": 0.0, "max": 250.0, "unit": "kg/m³", "reference": 125.0},
    "Fiber": {"min": 0.0, "max": 0.30, "unit": "%", "reference": 0.15},
}

DEFAULT_INPUTS = {
    "Water": 350.0,
    "Cement": 825.0,
    "SCM_Type": "Quartz",
    "SCM_Amount": 125.0,
    "Fiber": 0.15,
}

SAMPLE_MIXES = {
    "Balanced Mix": {
        "Water": 350.0,
        "Cement": 825.0,
        "SCM_Type": "Quartz",
        "SCM_Amount": 125.0,
        "Fiber": 0.15,
    },
    "Low Cement Mix": {
        "Water": 340.0,
        "Cement": 760.0,
        "SCM_Type": "Fly Ash",
        "SCM_Amount": 180.0,
        "Fiber": 0.18,
    },
    "High Strength Trial": {
        "Water": 320.0,
        "Cement": 950.0,
        "SCM_Type": "Silica Fume",
        "SCM_Amount": 120.0,
        "Fiber": 0.25,
    },
}

RELIABLE_MODEL_TYPES = {
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Gradient Boosting",
    "Decision Tree",
}

STRENGTH_COLUMN_CANDIDATES = [
    "Predicted_Strength",
    "Predicted Strength",
    "Prediction",
    "Prediction (MPa)",
    "Tensile_Strength",
    "Tensile Strength",
    "Strength",
    "target",
]


# ------------------------- Page setup -------------------------
st.set_page_config(
    page_title="Scenario B Tensile Strength Predictor",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 5.25rem !important;
        padding-bottom: 2rem !important;
        max-width: 1500px;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.94);
        backdrop-filter: blur(8px);
    }

    .app-title {
        font-size: 2.08rem;
        font-weight: 850;
        line-height: 1.18;
        margin-top: 0.45rem;
        margin-bottom: 0.20rem;
        color: #0f172a;
    }

    .app-subtitle {
        font-size: 1.02rem;
        color: #64748b;
        margin-bottom: 1.05rem;
    }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.65rem;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        color: #3730a3;
        font-size: 0.82rem;
        font-weight: 750;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
    }

    .metric-card {
        padding: 1.05rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(15, 23, 42, 0.09);
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        box-shadow: 0 8px 28px rgba(15,23,42,0.06);
        min-height: 128px;
    }

    .metric-label {
        font-size: 0.84rem;
        color: #64748b;
        font-weight: 750;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 2.05rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
    }

    .metric-note {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    .warning-box, .success-box, .info-box {
        padding: 0.85rem 0.95rem;
        border-radius: 14px;
        margin: 0.35rem 0 0.55rem 0;
        font-size: 0.95rem;
        line-height: 1.45;
    }

    .warning-box {background: #fffbeb; border: 1px solid #fde68a; color: #92400e;}
    .success-box {background: #ecfdf5; border: 1px solid #a7f3d0; color: #065f46;}
    .info-box {background: #eff6ff; border: 1px solid #bfdbfe; color: #1e3a8a;}

    .recommend-card {
        padding: 1.1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
        border: 1px solid #c7d2fe;
        box-shadow: 0 10px 30px rgba(30, 41, 59, 0.08);
        margin-bottom: 1rem;
    }

    .recommend-title {
        font-size: 1.08rem;
        font-weight: 850;
        color: #1e1b4b;
        margin-bottom: 0.35rem;
    }

    .recommend-body {
        font-size: 0.95rem;
        color: #334155;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------- Session state -------------------------
def init_input_state() -> None:
    for key, value in DEFAULT_INPUTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_sample_mix(sample_name: str) -> None:
    sample = SAMPLE_MIXES[sample_name]
    for key, value in sample.items():
        st.session_state[key] = value


init_input_state()


# ------------------------- Loading helpers -------------------------
def infer_model_type(info: Dict[str, Any]) -> str:
    existing = str(info.get("type", "")).strip()
    if existing and existing != "-":
        return existing

    name = str(info.get("name", "")).lower()
    file = str(info.get("file", "")).lower()
    text = f"{name} {file}"

    if "random forest" in text or "rf" in text:
        return "Random Forest"
    if "xgboost" in text or "xgb" in text:
        return "XGBoost"
    if "lightgbm" in text or "lgb" in text:
        return "LightGBM"
    if "catboost" in text or "cb" in text:
        return "CatBoost"
    if "gradient" in text or "gbr" in text:
        return "Gradient Boosting"
    if "decision" in text or "dt" in text:
        return "Decision Tree"
    if "ann" in text or "mlp" in text:
        return "ANN"
    if "ridge" in text:
        return "Ridge"
    if "lasso" in text:
        return "Lasso"
    if "linear" in text:
        return "Linear Regression"
    if "svr" in text:
        return "SVR"
    return "Other"


@st.cache_data
def load_registry() -> List[Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)
    else:
        registry = [
            {
                "name": "Random Forest Tuned — Scenario B",
                "file": "models/tuned_rf.pkl",
                "type": "Random Forest",
                "status": "Primary / Best",
                "show_in_comparison": True,
            },
            {
                "name": "Random Forest Default — Scenario B",
                "file": "models/rf.pkl",
                "type": "Random Forest",
                "status": "Comparison",
                "show_in_comparison": True,
            },
            {
                "name": "XGBoost Default — Scenario B",
                "file": "models/xgb.pkl",
                "type": "XGBoost",
                "status": "Comparison",
                "show_in_comparison": True,
            },
            {
                "name": "LightGBM Default — Scenario B",
                "file": "models/lgb.pkl",
                "type": "LightGBM",
                "status": "Comparison",
                "show_in_comparison": True,
            },
            {
                "name": "CatBoost Default — Scenario B",
                "file": "models/cb.pkl",
                "type": "CatBoost",
                "status": "Comparison",
                "show_in_comparison": True,
            },
        ]

    normalized = []
    for item in registry:
        item = dict(item)
        item.setdefault("name", "Unnamed model")
        item.setdefault("file", "")
        item["type"] = infer_model_type(item)
        item.setdefault("status", "Comparison")
        item.setdefault("show_in_comparison", True)
        item.setdefault("r2", None)
        item.setdefault("mae", None)
        item.setdefault("rmse", None)

        if item["name"] == PRIMARY_MODEL_NAME:
            item["status"] = "Primary / Best"

        normalized.append(item)

    normalized.sort(key=lambda x: 0 if x.get("name") == PRIMARY_MODEL_NAME else 1)
    return normalized


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
        if matched is not None:
            df["Predicted_Strength"] = pd.to_numeric(df[matched], errors="coerce")
        else:
            df["Predicted_Strength"] = np.nan
    else:
        df["Predicted_Strength"] = pd.to_numeric(df["Predicted_Strength"], errors="coerce")

    if "Active_SCM" not in df.columns:
        df["Active_SCM"] = df[SCM_COLUMNS].idxmax(axis=1)

    df["Active_SCM"] = df["Active_SCM"].astype(str)
    df = df.dropna(subset=["Predicted_Strength"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def load_model(model_file: str) -> Tuple[Any, str]:
    model_path = APP_DIR / model_file
    if not model_path.exists():
        return None, f"File not found: {model_file}"

    if joblib is not None:
        try:
            return joblib.load(model_path), "Loaded"
        except Exception as joblib_error:
            joblib_msg = str(joblib_error)
    else:
        joblib_msg = "joblib is not installed"

    try:
        with open(model_path, "rb") as f:
            return pickle.load(f), "Loaded"
    except Exception as pickle_error:
        return None, f"Could not load {model_file}. joblib: {joblib_msg}; pickle: {pickle_error}"


# ------------------------- Input conversion -------------------------
def build_scenario_b_input(
    water: float,
    cement: float,
    scm_type: str,
    scm_amount: float,
    fiber: float,
) -> pd.DataFrame:
    row = {col: 0.0 for col in FEATURE_COLUMNS}
    row["Water"] = float(water)
    row["Cement"] = float(cement)
    row["Fiber"] = float(fiber)

    if scm_type in SCM_COLUMNS:
        row[scm_type] = float(scm_amount)

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def row_to_ui_values(row: pd.DataFrame) -> Dict[str, Any]:
    active_scm = max(SCM_COLUMNS, key=lambda c: float(row.loc[0, c]))
    return {
        "Water": float(row.loc[0, "Water"]),
        "Cement": float(row.loc[0, "Cement"]),
        "SCM_Type": active_scm,
        "SCM_Amount": float(row.loc[0, active_scm]),
        "Fiber": float(row.loc[0, "Fiber"]),
    }

def is_reasonable_tensile_strength(value: float) -> bool:
    try:
        v = float(value)
    except Exception:
        return False

    if not np.isfinite(v):
        return False

    return 0.0 <= v <= 20.0


def predict_with_model(row: pd.DataFrame, model_info: Dict[str, Any]) -> Tuple[float, str]:
    """
    Predict only using the provided trained model file.

    No demo fallback.
    No random prediction.
    No manually generated prediction.

    If the model cannot be loaded or cannot predict, the function returns NaN
    and the GUI will show an error.
    """
    model_file = model_info.get("file", "")
    model, load_status = load_model(model_file)

    if model is None:
        return np.nan, f"Model not loaded: {load_status}"

    x = row[FEATURE_COLUMNS].copy()

    try:
        pred = model.predict(x)[0]
        pred = float(np.ravel(pred)[0])
    except Exception as dataframe_error:
        try:
            pred = model.predict(x.to_numpy())[0]
            pred = float(np.ravel(pred)[0])
        except Exception as array_error:
            return np.nan, (
                "Prediction failed using provided model. "
                f"DataFrame error: {dataframe_error}; "
                f"Array error: {array_error}"
            )

    if not np.isfinite(pred):
        return np.nan, "Provided model returned a non-finite prediction."

    if not is_reasonable_tensile_strength(pred):
        return np.nan, (
            f"Provided model returned an unrealistic value: {pred:.6g}. "
            "This usually means the model needs its original scaler/preprocessing pipeline."
        )

    return round(pred, 4), "Real provided Scenario B model"


def extract_tree_ensemble(model: Any) -> Optional[Any]:
    if model is None:
        return None

    if hasattr(model, "estimators_"):
        return model

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "estimators_"):
                return step

    if hasattr(model, "steps"):
        for _, step in reversed(list(model.steps)):
            if hasattr(step, "estimators_"):
                return step

    return None


def rf_tree_uncertainty(row: pd.DataFrame, model_info: Dict[str, Any]) -> Tuple[Optional[float], int, str]:
    model, load_status = load_model(model_info.get("file", ""))

    if model is None:
        return None, 0, f"Model not loaded: {load_status}"

    ensemble = extract_tree_ensemble(model)

    if ensemble is None or not hasattr(ensemble, "estimators_"):
        return None, 0, "Tree-level uncertainty is available only for tree ensembles."

    x_df = row[FEATURE_COLUMNS].copy()
    x_np = x_df.to_numpy()
    tree_preds = []

    for estimator in ensemble.estimators_:
        try:
            p = estimator.predict(x_df)[0]
        except Exception:
            try:
                p = estimator.predict(x_np)[0]
            except Exception:
                continue

        try:
            p_float = float(np.ravel(p)[0])
            if is_reasonable_tensile_strength(p_float):
                tree_preds.append(p_float)
        except Exception:
            continue

    if len(tree_preds) < 2:
        return None, len(tree_preds), "Not enough valid tree predictions from the provided RF model."

    return float(np.std(tree_preds)), len(tree_preds), "RF tree-based uncertainty from provided model"


def check_training_ranges(ui_values: Dict[str, Any]) -> List[str]:
    warnings = []

    for key in ["Water", "Cement", "SCM_Amount", "Fiber"]:
        value = float(ui_values[key])
        limits = TRAINING_RANGES[key]

        if value < limits["min"]:
            warnings.append(
                f"{key.replace('_', ' ')} value ({value:g} {limits['unit']}) is below training range "
                f"({limits['min']:g}–{limits['max']:g} {limits['unit']})."
            )
        elif value > limits["max"]:
            warnings.append(
                f"{key.replace('_', ' ')} value ({value:g} {limits['unit']}) exceeds training range "
                f"({limits['min']:g}–{limits['max']:g} {limits['unit']})."
            )

    return warnings


def calculate_reliability_score(ui_values: Dict[str, Any]) -> Tuple[int, str, List[str]]:
    score = 100.0
    notes = []

    for key in ["Water", "Cement", "SCM_Amount", "Fiber"]:
        value = float(ui_values[key])
        limits = TRAINING_RANGES[key]
        mn = float(limits["min"])
        mx = float(limits["max"])
        width = mx - mn
        scaled = (value - mn) / width if width else 0.5

        if value < mn or value > mx:
            distance = abs(value - mn) if value < mn else abs(value - mx)
            penalty = 30.0 + min(35.0, 100.0 * distance / max(width, 1e-9))
            score -= penalty
            notes.append(f"{key.replace('_', ' ')} is outside the training range.")
        elif scaled <= 0.05 or scaled >= 0.95:
            score -= 10.0
            notes.append(f"{key.replace('_', ' ')} is very close to a training-range boundary.")
        elif scaled <= 0.15 or scaled >= 0.85:
            score -= 5.0
            notes.append(f"{key.replace('_', ' ')} is near a training-range boundary.")

    score_int = int(round(max(0.0, min(100.0, score))))

    if score_int >= 85:
        label = "High"
    elif score_int >= 60:
        label = "Medium"
    else:
        label = "Low"

    if not notes:
        notes.append("All inputs are comfortably inside the Scenario B training domain.")

    return score_int, label, notes


def metric_or_dash(model_info: Dict[str, Any], key: str) -> Any:
    value = model_info.get(key, None)

    if value is None or value == "":
        return "—"

    try:
        return round(float(value), 4)
    except Exception:
        return value


def prediction_comparison(
    row: pd.DataFrame,
    registry: List[Dict[str, Any]],
    primary_model_info: Dict[str, Any],
) -> pd.DataFrame:
    records = []

    primary_pred, primary_mode = predict_with_model(row, primary_model_info)

    for info in registry:
        if not info.get("show_in_comparison", True):
            continue

        model_type = infer_model_type(info)

        if model_type not in RELIABLE_MODEL_TYPES:
            continue

        pred, mode = predict_with_model(row, info)

        if not np.isfinite(pred):
            continue

        records.append(
            {
                "Model": info.get("name", "Unnamed model"),
                "Type": model_type,
                "Prediction (MPa)": round(float(pred), 4),
                "Difference from RF Tuned": (
                    round(float(pred - primary_pred), 4)
                    if np.isfinite(primary_pred)
                    else "—"
                ),
                "R²": metric_or_dash(info, "r2"),
                "MAE": metric_or_dash(info, "mae"),
                "RMSE": metric_or_dash(info, "rmse"),
                "Status": info.get("status", "Comparison"),
                "Mode": mode,
                "PrimarySort": 0 if info.get("name") == PRIMARY_MODEL_NAME else 1,
            }
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "Model",
                "Type",
                "Prediction (MPa)",
                "Difference from RF Tuned",
                "R²",
                "MAE",
                "RMSE",
                "Status",
                "Mode",
            ]
        )

    df = pd.DataFrame(records)
    df = df.sort_values(
        ["PrimarySort", "Prediction (MPa)"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return df.drop(columns=["PrimarySort"], errors="ignore")

# ------------------------- Explanation helpers -------------------------
def local_sensitivity_contributions(
    row: pd.DataFrame,
    selected_model_info: Dict[str, Any],
    selected_prediction: float,
) -> pd.DataFrame:
    ui = row_to_ui_values(row)
    current = selected_prediction
    contributions = []

    reference_settings = [
        ("Water", TRAINING_RANGES["Water"]["reference"], "Compares current water with reference water content."),
        ("Cement", TRAINING_RANGES["Cement"]["reference"], "Compares current cement with reference cement content."),
        ("SCM_Amount", TRAINING_RANGES["SCM_Amount"]["reference"], "Compares selected SCM dosage with reference dosage."),
        ("Fiber", TRAINING_RANGES["Fiber"]["reference"], "Compares fiber content with reference fiber content."),
    ]

    for feature, ref_value, note in reference_settings:
        altered = dict(ui)
        altered[feature] = ref_value

        altered_row = build_scenario_b_input(
            altered["Water"],
            altered["Cement"],
            altered["SCM_Type"],
            altered["SCM_Amount"],
            altered["Fiber"],
        )

        ref_pred, _ = predict_with_model(altered_row, selected_model_info)
        contribution = current - ref_pred

        contributions.append(
            {
                "Feature": feature.replace("_", " "),
                "Current Value": ui[feature],
                "Reference": ref_value,
                "Contribution (MPa)": round(float(contribution), 4),
                "Interpretation": note,
            }
        )

    if ui["SCM_Type"] != "Quartz":
        quartz_row = build_scenario_b_input(
            ui["Water"], ui["Cement"], "Quartz", ui["SCM_Amount"], ui["Fiber"]
        )
        quartz_pred, _ = predict_with_model(quartz_row, selected_model_info)
        scm_type_contribution = current - quartz_pred
        reference_label = "Quartz"
    else:
        zero_scm_row = build_scenario_b_input(
            ui["Water"], ui["Cement"], "Quartz", 0.0, ui["Fiber"]
        )
        zero_scm_pred, _ = predict_with_model(zero_scm_row, selected_model_info)
        scm_type_contribution = current - zero_scm_pred
        reference_label = "Quartz amount = 0"

    contributions.append(
        {
            "Feature": "SCM Type",
            "Current Value": ui["SCM_Type"],
            "Reference": reference_label,
            "Contribution (MPa)": round(float(scm_type_contribution), 4),
            "Interpretation": "Material-specific effect of the selected SCM.",
        }
    )

    df = pd.DataFrame(contributions)
    return df.reindex(df["Contribution (MPa)"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def make_contribution_chart(contrib_df: pd.DataFrame) -> go.Figure:
    plot_df = contrib_df.sort_values("Contribution (MPa)", ascending=True).copy()
    max_abs = max(0.05, float(plot_df["Contribution (MPa)"].abs().max()))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=plot_df["Contribution (MPa)"],
            y=plot_df["Feature"],
            orientation="h",
            marker=dict(
                color=plot_df["Contribution (MPa)"],
                colorscale=[
                    [0.0, "#dc2626"],
                    [0.5, "#e5e7eb"],
                    [1.0, "#16a34a"],
                ],
                cmin=-max_abs,
                cmax=max_abs,
                line=dict(color="rgba(15,23,42,0.20)", width=1),
            ),
            text=[f"{v:+.3f} MPa" for v in plot_df["Contribution (MPa)"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f} MPa<extra></extra>",
            cliponaxis=False,
        )
    )

    fig.add_vline(x=0, line_width=1.6, line_dash="dash", line_color="#0f172a")

    fig.update_layout(
        title="Local Explanation: Contribution of Each Input",
        xaxis_title="Change in predicted tensile strength (MPa)",
        yaxis_title="",
        height=410,
        template="plotly_white",
        margin=dict(l=20, r=95, t=60, b=40),
        xaxis=dict(range=[-1.35 * max_abs, 1.35 * max_abs], zeroline=False),
        font=dict(size=13),
    )

    return fig


def make_profile_radar(ui: Dict[str, Any]) -> go.Figure:
    labels = ["Water", "Cement", "SCM Amount", "Fiber"]
    values = []

    for key in ["Water", "Cement", "SCM_Amount", "Fiber"]:
        limits = TRAINING_RANGES[key]
        val = float(ui[key])
        scaled = (val - limits["min"]) / (limits["max"] - limits["min"])
        values.append(max(0, min(1, scaled)))

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name="Normalized mix profile",
            line_color="#2563eb",
        )
    )

    fig.update_layout(
        title="Mix Position within Training Range",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0, 0.5, 1],
                ticktext=["Min", "Mid", "Max"],
            )
        ),
        showlegend=False,
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=55, b=25),
    )

    return fig


def make_reliability_gauge(score: int, label: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={
                "text": f"Input Reliability Score<br><span style='font-size:0.8em;color:#64748b'>{label} reliability</span>"
            },
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 60], "color": "#fee2e2"},
                    {"range": [60, 85], "color": "#fef3c7"},
                    {"range": [85, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "#0f172a", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )

    fig.update_layout(
        height=280,
        template="plotly_white",
        margin=dict(l=20, r=20, t=50, b=15),
    )

    return fig


def make_waterfall_chart(contrib_df: pd.DataFrame, selected_prediction: float) -> go.Figure:
    plot_df = contrib_df.copy().sort_values(
        "Contribution (MPa)",
        key=lambda s: s.abs(),
        ascending=False,
    )

    total_effect = float(plot_df["Contribution (MPa)"].sum())
    reference_baseline = float(selected_prediction - total_effect)

    fig = go.Figure(
        go.Waterfall(
            name="Local explanation",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(plot_df) + ["total"],
            x=["Reference baseline"] + plot_df["Feature"].tolist() + ["Current prediction"],
            y=[reference_baseline] + plot_df["Contribution (MPa)"].tolist() + [selected_prediction],
            text=[f"{reference_baseline:.3f}"]
            + [f"{v:+.3f}" for v in plot_df["Contribution (MPa)"]]
            + [f"{selected_prediction:.3f}"],
            textposition="outside",
            connector={"line": {"color": "#94a3b8"}},
        )
    )

    fig.update_layout(
        title="Approximate Local Waterfall Explanation",
        yaxis_title="Predicted tensile strength (MPa)",
        height=430,
        template="plotly_white",
        margin=dict(l=15, r=25, t=60, b=80),
    )

    return fig


def make_what_if_curve(
    row: pd.DataFrame,
    model_info: Dict[str, Any],
    feature: str,
) -> pd.DataFrame:
    ui = row_to_ui_values(row)
    limits = TRAINING_RANGES[feature]
    values = np.linspace(limits["min"], limits["max"], 45)
    records = []

    for v in values:
        changed = dict(ui)
        changed[feature] = float(v)

        changed_row = build_scenario_b_input(
            changed["Water"],
            changed["Cement"],
            changed["SCM_Type"],
            changed["SCM_Amount"],
            changed["Fiber"],
        )

        pred, _ = predict_with_model(changed_row, model_info)
        records.append(
            {
                feature.replace("_", " "): float(v),
                "Predicted Strength (MPa)": pred,
            }
        )

    return pd.DataFrame(records)


def make_scm_choice_chart(row: pd.DataFrame, model_info: Dict[str, Any]) -> pd.DataFrame:
    ui = row_to_ui_values(row)
    records = []

    for scm in SCM_COLUMNS:
        candidate = build_scenario_b_input(
            ui["Water"],
            ui["Cement"],
            scm,
            ui["SCM_Amount"],
            ui["Fiber"],
        )
        pred, _ = predict_with_model(candidate, model_info)
        records.append({"SCM Type": scm, "Predicted Strength (MPa)": pred})

    return pd.DataFrame(records).sort_values("Predicted Strength (MPa)", ascending=False)


def generate_explanation_text(contrib_df: pd.DataFrame, selected_prediction: float) -> str:
    top_positive = contrib_df[contrib_df["Contribution (MPa)"] > 0].head(2)
    top_negative = contrib_df[contrib_df["Contribution (MPa)"] < 0].head(2)

    parts = [
        f"The selected model predicts a tensile strength of **{selected_prediction:.3f} MPa** for the current mix.",
        "The local explanation compares the current input with reference values from the Scenario B training domain.",
    ]

    if not top_positive.empty:
        pos_text = ", ".join(
            [f"{r['Feature']} ({r['Contribution (MPa)']:+.3f} MPa)" for _, r in top_positive.iterrows()]
        )
        parts.append(f"The strongest positive contributors are: **{pos_text}**.")

    if not top_negative.empty:
        neg_text = ", ".join(
            [f"{r['Feature']} ({r['Contribution (MPa)']:+.3f} MPa)" for _, r in top_negative.iterrows()]
        )
        parts.append(f"The strongest reducing contributors are: **{neg_text}**.")

    parts.append("For final engineering use, the predicted mix should still be validated experimentally.")

    return " ".join(parts)


# ------------------------- Optimizer helpers -------------------------
def score_optimizer_candidates(df: pd.DataFrame, goal: str) -> pd.DataFrame:
    out = df.copy()

    if goal == "Maximum strength":
        out["Score"] = out["Predicted_Strength"]
        return out.sort_values("Score", ascending=False)

    if goal == "Minimum cement":
        out["Score"] = -out["Cement"]
        return out.sort_values(["Score", "Predicted_Strength"], ascending=[False, False])

    if goal == "Minimum fiber":
        out["Score"] = -out["Fiber"]
        return out.sort_values(["Score", "Predicted_Strength"], ascending=[False, False])

    def normalize(series: pd.Series) -> pd.Series:
        if series.max() == series.min():
            return pd.Series(np.ones(len(series)), index=series.index)
        return (series - series.min()) / (series.max() - series.min())

    s_strength = normalize(out["Predicted_Strength"])
    s_cement = normalize(out["Cement"])
    s_water = normalize(out["Water"])
    s_fiber = normalize(out["Fiber"])

    out["Score"] = 0.55 * s_strength - 0.22 * s_cement - 0.13 * s_water - 0.10 * s_fiber
    return out.sort_values("Score", ascending=False)


def filter_pareto_candidates(
    pareto: pd.DataFrame,
    scm_filter: str,
    max_cement: float,
    max_water: float,
    max_fiber: float,
    min_strength: float,
    goal: str,
) -> pd.DataFrame:
    if pareto.empty:
        return pareto

    df = pareto.copy()

    if scm_filter != "Any SCM":
        df = df[df["Active_SCM"].astype(str) == scm_filter]

    df = df[
        (df["Cement"] <= max_cement)
        & (df["Water"] <= max_water)
        & (df["Fiber"] <= max_fiber)
        & (df["Predicted_Strength"] >= min_strength)
    ]

    if df.empty:
        return df

    return score_optimizer_candidates(df, goal).reset_index(drop=True)


# ------------------------- Report helper -------------------------
def generate_prediction_report(
    ui_values: Dict[str, Any],
    selected_model_name: str,
    selected_prediction: float,
    selected_mode: str,
    uncertainty_text: str,
    reliability_score: int,
    reliability_label: str,
    warnings: List[str],
    compare_df: pd.DataFrame,
) -> str:
    lines = [
        "Scenario B Tensile Strength Prediction Report",
        "=" * 52,
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Input Mix Design",
        "-" * 52,
        f"Water: {ui_values['Water']:.3f} kg/m³",
        f"Cement: {ui_values['Cement']:.3f} kg/m³",
        f"SCM Type: {ui_values['SCM_Type']}",
        f"SCM Amount: {ui_values['SCM_Amount']:.3f} kg/m³",
        f"Fiber: {ui_values['Fiber']:.3f} %",
        "",
        "Prediction Output",
        "-" * 52,
        f"Selected model: {selected_model_name}",
        f"Prediction mode: {selected_mode}",
        f"Predicted tensile strength: {selected_prediction:.4f} MPa",
        f"Uncertainty / agreement: {uncertainty_text}",
        f"Input reliability score: {reliability_score}% ({reliability_label})",
        "",
        "Training Range Status",
        "-" * 52,
    ]

    if warnings:
        for w in warnings:
            lines.append(f"WARNING: {w}")
    else:
        lines.append("All entered values are within the Scenario B training range.")

    lines.extend(["", "Quick Model Comparison", "-" * 52])

    if compare_df.empty:
        lines.append("No comparison data available.")
    else:
        export_cols = ["Model", "Prediction (MPa)", "Difference from RF Tuned", "Status", "Mode"]
        export_df = compare_df[[c for c in export_cols if c in compare_df.columns]].copy()
        lines.append(export_df.to_string(index=False))

    lines.extend(
        [
            "",
            "Engineering Note",
            "-" * 52,
            "This output is generated by a machine-learning model and should be verified experimentally before practical mix-design adoption.",
        ]
    )

    return "\n".join(lines)


# ------------------------- Load resources -------------------------
registry = load_registry()
pareto_df = load_pareto_data()

primary_model = next((m for m in registry if m.get("name") == PRIMARY_MODEL_NAME), registry[0])
model_names = [m["name"] for m in registry]


# ------------------------- Sidebar -------------------------
with st.sidebar:
    st.markdown("### Navigation")
    section = st.selectbox("Choose section", ["Predictor", "Explanation", "Optimizer"], index=0)

    st.markdown("### Model")
    selected_model_name = st.selectbox(
        "Prediction model",
        model_names,
        index=model_names.index(PRIMARY_MODEL_NAME) if PRIMARY_MODEL_NAME in model_names else 0,
        help="Scenario B Random Forest Tuned is kept as the default/best model.",
    )
    selected_model_info = next(m for m in registry if m["name"] == selected_model_name)

    st.markdown("### Recommended Model")
    st.caption("⭐ Random Forest Tuned — Scenario B")
    st.caption("Use other models only for comparison or sensitivity checking.")

    st.markdown("### Scenario B Features")
    st.caption("Water, Cement, Quartz, Fly Ash, Bagasse, Silica Fume, Calcium Carbonate, Fiber")

    st.markdown("### Valid Training Ranges")
    for k, v in TRAINING_RANGES.items():
        st.caption(f"{k.replace('_', ' ')}: {v['min']:g}–{v['max']:g} {v['unit']}")


# ------------------------- Shared header and input -------------------------
st.markdown('<div class="app-title">Scenario B Tensile Strength Predictor</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="app-subtitle">Web-based GUI using the best model: '
    '<b>Random Forest Tuned — Scenario B</b>. The app converts SCM type and dosage '
    'into the exact one-hot Scenario B input format automatically.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="pill">Primary model: RF Tuned — Scenario B</span>'
    '<span class="pill">Prediction dashboard</span>'
    '<span class="pill">Model comparison kept</span>'
    '<span class="pill">Reliability score</span>'
    '<span class="pill">Pareto optimizer included</span>',
    unsafe_allow_html=True,
)

with st.expander("Input mix design", expanded=True):
    st.markdown("**Quick samples**")

    s1, s2, s3, s4 = st.columns([1, 1, 1, 2])

    with s1:
        if st.button("Balanced Mix", use_container_width=True):
            apply_sample_mix("Balanced Mix")

    with s2:
        if st.button("Low Cement Mix", use_container_width=True):
            apply_sample_mix("Low Cement Mix")

    with s3:
        if st.button("High Strength Trial", use_container_width=True):
            apply_sample_mix("High Strength Trial")

    with s4:
        st.caption("Use these buttons during presentation/demo to quickly load meaningful examples.")

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.25, 1, 1])

    with c1:
        water = st.number_input(
            "Water (kg/m³)",
            min_value=250.0,
            max_value=450.0,
            step=1.0,
            key="Water",
        )

    with c2:
        cement = st.number_input(
            "Cement (kg/m³)",
            min_value=600.0,
            max_value=1200.0,
            step=1.0,
            key="Cement",
        )

    with c3:
        if st.session_state.get("SCM_Type") not in SCM_TYPES:
            st.session_state["SCM_Type"] = SCM_TYPES[0]

        scm_type = st.selectbox("SCM Type", SCM_TYPES, key="SCM_Type")

    with c4:
        scm_amount = st.number_input(
            "SCM Amount (kg/m³)",
            min_value=0.0,
            max_value=350.0,
            step=1.0,
            key="SCM_Amount",
        )

    with c5:
        fiber = st.number_input(
            "Fiber (%)",
            min_value=0.0,
            max_value=0.50,
            step=0.01,
            format="%.3f",
            key="Fiber",
        )


# ------------------------- Shared computation -------------------------
row = build_scenario_b_input(water, cement, scm_type, scm_amount, fiber)
ui_values = row_to_ui_values(row)

warnings = check_training_ranges(ui_values)
reliability_score, reliability_label, reliability_notes = calculate_reliability_score(ui_values)

selected_prediction, selected_mode = predict_with_model(row, selected_model_info)
primary_prediction, primary_mode = predict_with_model(row, primary_model)

compare_df = prediction_comparison(row, registry, primary_model)

if not np.isfinite(selected_prediction):
    st.error(
        "The selected model could not produce a valid prediction. "
        "No demo or random fallback value is being used."
    )
    st.code(selected_mode)

    st.warning(
        "Most likely reason: the model was trained with a different scikit-learn "
        "version or it requires the original scaler/preprocessing pipeline."
    )

    st.stop()

if not np.isfinite(primary_prediction):
    st.error(
        "The primary Random Forest Tuned — Scenario B model could not produce "
        "a valid prediction. Please check the model file and environment."
    )
    st.code(primary_mode)
    st.stop()

valid_compare_df = compare_df[compare_df["Prediction (MPa)"].apply(is_reasonable_tensile_strength)].copy()

if len(valid_compare_df) > 1:
    model_spread = float(valid_compare_df["Prediction (MPa)"].std())
    min_pred = float(valid_compare_df["Prediction (MPa)"].min())
    max_pred = float(valid_compare_df["Prediction (MPa)"].max())
else:
    model_spread = 0.0
    min_pred = max_pred = float(selected_prediction)

rf_std, rf_tree_count, rf_uncertainty_mode = rf_tree_uncertainty(row, primary_model)

if rf_std is not None:
    uncertainty_value = rf_std
    uncertainty_note = f"RF tree-based uncertainty using {rf_tree_count} trees"
else:
    uncertainty_value = model_spread
    uncertainty_note = "Model agreement spread from reliable comparison models"

uncertainty_text = f"±{uncertainty_value:.3f} MPa ({uncertainty_note})"
input_status = "Within range" if not warnings else "Check warnings"

if selected_model_name != PRIMARY_MODEL_NAME:
    st.markdown(
        f'<div class="warning-box">⚠ You selected <b>{selected_model_name}</b>. '
        f'The recommended/default model for final reporting is still <b>{PRIMARY_MODEL_NAME}</b>.</div>',
        unsafe_allow_html=True,
    )


# ------------------------- Predictor section -------------------------
if section == "Predictor":
    card1, card2, card3, card4 = st.columns(4)

    with card1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Predicted tensile strength</div>
                <div class="metric-value">{selected_prediction:.3f} MPa</div>
                <div class="metric-note">Selected model: {selected_model_name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Uncertainty / agreement</div>
                <div class="metric-value">±{uncertainty_value:.3f}</div>
                <div class="metric-note">{uncertainty_note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Input reliability</div>
                <div class="metric-value">{reliability_score}%</div>
                <div class="metric-note">Reliability level: {reliability_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with card4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Training domain status</div>
                <div class="metric-value" style="font-size:1.65rem;">{input_status}</div>
                <div class="metric-note">Scenario B valid range checking is active.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns([1.05, 1.35], gap="large")

    with left:
        st.markdown("#### Recommended model result")

        st.markdown(
            f"""
            <div class="recommend-card">
                <div class="recommend-title">⭐ {PRIMARY_MODEL_NAME}</div>
                <div class="recommend-body">
                    Primary RF prediction: <b>{primary_prediction:.3f} MPa</b><br>
                    Selected-model prediction: <b>{selected_prediction:.3f} MPa</b><br>
                    Mode: {selected_mode}<br>
                    Reliability: {reliability_score}% ({reliability_label})
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if warnings:
            for msg in warnings:
                st.markdown(
                    f'<div class="warning-box">⚠ {msg}<br>'
                    f"Prediction is outside the training domain and should be treated as extrapolation.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="success-box">✓ All entered values are within the Scenario B training range.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Reliability notes")
        for note in reliability_notes:
            st.caption(f"• {note}")

        st.markdown("#### Scenario B input sent to model")
        st.dataframe(row, use_container_width=True, hide_index=True)

        report_text = generate_prediction_report(
            ui_values=ui_values,
            selected_model_name=selected_model_name,
            selected_prediction=selected_prediction,
            selected_mode=selected_mode,
            uncertainty_text=uncertainty_text,
            reliability_score=reliability_score,
            reliability_label=reliability_label,
            warnings=warnings,
            compare_df=compare_df,
        )

        st.download_button(
            "Download prediction report",
            data=report_text.encode("utf-8"),
            file_name="scenario_b_prediction_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with right:
        st.markdown("#### Quick Model Comparison")

        table = compare_df[
            [
                "Model",
                "Type",
                "Prediction (MPa)",
                "Difference from RF Tuned",
                "R²",
                "MAE",
                "Status",
                "Mode",
            ]
        ].copy()

        st.dataframe(table, use_container_width=True, hide_index=True)

        if not valid_compare_df.empty:
            fig = px.bar(
                valid_compare_df.sort_values("Prediction (MPa)", ascending=True),
                x="Prediction (MPa)",
                y="Model",
                orientation="h",
                color="Status",
                title="Quick Comparison of Reliable Scenario B Models",
                text="Prediction (MPa)",
                template="plotly_white",
            )

            fig.update_traces(
                texttemplate="%{text:.3f} MPa",
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                height=460,
                margin=dict(l=10, r=90, t=60, b=35),
                yaxis_title="",
                xaxis_title="Predicted tensile strength (MPa)",
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No reliable comparison models are available for this input.")


# ------------------------- Explanation section -------------------------
elif section == "Explanation":
    st.markdown("#### Explanation and Visual Diagnosis")

    st.markdown(
        '<div class="info-box">This section gives a practical local explanation. It compares the current mix with reference values and shows which input is pushing the prediction upward or downward. Later, this can be replaced with exact SHAP values if SHAP background data is saved.</div>',
        unsafe_allow_html=True,
    )

    contrib_df = local_sensitivity_contributions(row, selected_model_info, selected_prediction)
    st.markdown(generate_explanation_text(contrib_df, selected_prediction))

    top1, top2 = st.columns([1.15, 1], gap="large")

    with top1:
        st.plotly_chart(make_contribution_chart(contrib_df), use_container_width=True)

    with top2:
        st.plotly_chart(make_profile_radar(ui_values), use_container_width=True)

    mid1, mid2 = st.columns([1.1, 1], gap="large")

    with mid1:
        st.plotly_chart(make_waterfall_chart(contrib_df, selected_prediction), use_container_width=True)

    with mid2:
        st.plotly_chart(make_reliability_gauge(reliability_score, reliability_label), use_container_width=True)

    st.markdown("#### Contribution Table")
    st.dataframe(contrib_df, use_container_width=True, hide_index=True)

    st.markdown("#### What-if Sensitivity Curve")

    curve_feature = st.selectbox(
        "Select one input to vary",
        ["Water", "Cement", "SCM_Amount", "Fiber"],
        index=3,
    )

    curve_df = make_what_if_curve(row, selected_model_info, curve_feature)
    x_col = curve_feature.replace("_", " ")

    fig_curve = px.line(
        curve_df,
        x=x_col,
        y="Predicted Strength (MPa)",
        markers=True,
        title=f"What-if curve: effect of {x_col} while other inputs remain fixed",
        template="plotly_white",
    )

    current_x = ui_values[curve_feature]
    fig_curve.add_vline(
        x=current_x,
        line_dash="dash",
        line_color="#0f172a",
        annotation_text="Current value",
    )

    fig_curve.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=55, b=35),
    )

    st.plotly_chart(fig_curve, use_container_width=True)

    scm_df = make_scm_choice_chart(row, selected_model_info)

    fig_scm = px.bar(
        scm_df,
        x="SCM Type",
        y="Predicted Strength (MPa)",
        text="Predicted Strength (MPa)",
        title="SCM Type Scenario Check at the Same Dosage",
        template="plotly_white",
    )

    fig_scm.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        cliponaxis=False,
    )

    fig_scm.update_layout(
        height=410,
        margin=dict(l=10, r=30, t=55, b=35),
    )

    st.plotly_chart(fig_scm, use_container_width=True)

    st.markdown("#### Model Agreement Band")

    if len(valid_compare_df) >= 2:
        mean_pred = valid_compare_df["Prediction (MPa)"].mean()
        min_model_pred = valid_compare_df["Prediction (MPa)"].min()
        max_model_pred = valid_compare_df["Prediction (MPa)"].max()

        fig_agreement = go.Figure()

        fig_agreement.add_trace(
            go.Indicator(
                mode="number+delta",
                value=selected_prediction,
                delta={"reference": mean_pred, "relative": False, "valueformat": ".3f"},
                title={
                    "text": (
                        "Selected model prediction<br>"
                        f"<span style='font-size:0.8em;color:#64748b'>"
                        f"Reliable model range: {min_model_pred:.3f}–{max_model_pred:.3f} MPa"
                        f"</span>"
                    )
                },
                number={"suffix": " MPa", "font": {"size": 42}},
            )
        )

        fig_agreement.update_layout(
            height=260,
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig_agreement, use_container_width=True)
    else:
        st.info("Model agreement band needs at least two reliable comparison models.")


# ------------------------- Optimizer section -------------------------
elif section == "Optimizer":
    st.markdown("#### Pareto-Based Optimal Mix Recommender")

    st.markdown(
        '<div class="info-box">This optimizer uses <b>data/combined_pareto_front.csv</b>. It filters feasible mixes and ranks them according to the selected design goal. The result is a decision-support recommendation, not a replacement for laboratory validation.</div>',
        unsafe_allow_html=True,
    )

    oc1, oc2, oc3, oc4, oc5 = st.columns([1, 1, 1, 1, 1.15])

    with oc1:
        opt_scm = st.selectbox("SCM filter", ["Any SCM"] + SCM_TYPES, index=0)

    with oc2:
        max_cement = st.number_input(
            "Max cement",
            min_value=750.0,
            max_value=1200.0,
            value=900.0,
            step=10.0,
        )

    with oc3:
        max_water = st.number_input(
            "Max water",
            min_value=300.0,
            max_value=450.0,
            value=380.0,
            step=5.0,
        )

    with oc4:
        max_fiber = st.number_input(
            "Max fiber",
            min_value=0.0,
            max_value=0.50,
            value=0.30,
            step=0.01,
            format="%.3f",
        )

    with oc5:
        min_strength = st.number_input(
            "Min strength (MPa)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.05,
        )

    goal = st.selectbox(
        "Optimization goal",
        ["Maximum strength", "Minimum cement", "Minimum fiber", "Balanced mix"],
        index=3,
    )

    results = filter_pareto_candidates(
        pareto_df,
        opt_scm,
        max_cement,
        max_water,
        max_fiber,
        min_strength,
        goal,
    )

    if pareto_df.empty:
        st.error(
            "Pareto file was not found or has no valid predicted-strength column. "
            "Keep data/combined_pareto_front.csv inside the project folder."
        )

    elif results.empty:
        st.warning(
            "No feasible mix was found for these constraints. "
            "Relax cement, water, fiber, SCM, or minimum strength limits."
        )

    else:
        best = results.iloc[0]

        st.markdown(
            f"""
            <div class="recommend-card">
                <div class="recommend-title">Recommended Mix Design</div>
                <div class="recommend-body">
                    <b>Water:</b> {best['Water']:.1f} kg/m³ &nbsp; | &nbsp;
                    <b>Cement:</b> {best['Cement']:.1f} kg/m³ &nbsp; | &nbsp;
                    <b>SCM:</b> {best['Active_SCM']} &nbsp; | &nbsp;
                    <b>Fiber:</b> {best['Fiber']:.3f} %<br>
                    <b>Predicted tensile strength:</b> {best['Predicted_Strength']:.3f} MPa<br>
                    <b>Reason:</b> This is the highest-ranked feasible mix for the selected goal: {goal}.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m1, m2, m3, m4, m5 = st.columns(5)

        m1.metric("Water", f"{best['Water']:.1f} kg/m³")
        m2.metric("Cement", f"{best['Cement']:.1f} kg/m³")
        m3.metric("SCM", str(best["Active_SCM"]))
        m4.metric("Fiber", f"{best['Fiber']:.3f} %")
        m5.metric("Strength", f"{best['Predicted_Strength']:.3f} MPa")

        display_cols = ["Water", "Cement"] + SCM_COLUMNS + ["Fiber", "Predicted_Strength", "Active_SCM", "Score"]
        display_cols = [c for c in display_cols if c in results.columns]

        st.markdown("#### Top feasible mixes")
        st.dataframe(results[display_cols].head(25), use_container_width=True, hide_index=True)

        p1, p2 = st.columns(2, gap="large")

        with p1:
            fig1 = px.scatter(
                results.head(300),
                x="Cement",
                y="Predicted_Strength",
                color="Active_SCM",
                size="Fiber",
                hover_data=["Water"] + SCM_COLUMNS,
                title="Cement vs Predicted Strength",
                template="plotly_white",
            )

            fig1.add_scatter(
                x=[best["Cement"]],
                y=[best["Predicted_Strength"]],
                mode="markers+text",
                text=["Recommended"],
                textposition="top center",
                marker=dict(size=16, color="#0f172a", symbol="star"),
                name="Recommended mix",
            )

            fig1.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=55, b=35),
            )

            st.plotly_chart(fig1, use_container_width=True)

        with p2:
            fig2 = px.scatter(
                results.head(300),
                x="Water",
                y="Predicted_Strength",
                color="Active_SCM",
                size="Cement",
                hover_data=["Fiber"] + SCM_COLUMNS,
                title="Water vs Predicted Strength",
                template="plotly_white",
            )

            fig2.add_scatter(
                x=[best["Water"]],
                y=[best["Predicted_Strength"]],
                mode="markers+text",
                text=["Recommended"],
                textposition="top center",
                marker=dict(size=16, color="#0f172a", symbol="star"),
                name="Recommended mix",
            )

            fig2.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=55, b=35),
            )

            st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.scatter(
            results.head(500),
            x="Cement",
            y="Predicted_Strength",
            color="Active_SCM",
            size="Fiber",
            facet_col="Active_SCM" if results["Active_SCM"].nunique() <= 5 else None,
            title="Pareto Feasible Region by SCM Type",
            template="plotly_white",
        )

        fig3.update_layout(
            height=460,
            margin=dict(l=10, r=10, t=60, b=35),
        )

        st.plotly_chart(fig3, use_container_width=True)

        csv = results[display_cols].to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download feasible optimized mixes",
            data=csv,
            file_name="optimized_scenario_b_mixes.csv",
            mime="text/csv",
            use_container_width=True,
        )