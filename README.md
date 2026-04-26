# Scenario B Tensile Strength Predictor — Streamlit Web App

This project is a browser-based GUI for your Scenario B tensile strength prediction models.

## Main design decision

- Main/best model: **Random Forest Tuned — Scenario B**
- Navigation style: **dropdown/sidebar section selector**, not many tabs
- Main sections:
  1. Predictor
  2. Explanation
  3. Optimizer
- Quick model comparison table is kept inside the Predictor section.

## Scenario B input format

The app automatically converts simple user inputs into the exact Scenario B model columns:

```text
Water, Cement, Quartz, Fly Ash, Bagasse, Silica Fume, Calcium Carbonate, Fiber
```

For example, if the user selects `Fly Ash` and SCM amount = `150`, the app sends:

```text
Fly Ash = 150
Quartz = 0
Bagasse = 0
Silica Fume = 0
Calcium Carbonate = 0
```

## Folder structure

```text
tensile_strength_scenarioB_web_app/
│
├── app.py
├── requirements.txt
├── runtime.txt
├── model_registry.json
│
├── models/
│   ├── tuned_rf.pkl
│   ├── rf.pkl
│   ├── tuned_xgb.pkl
│   ├── xgb.pkl
│   ├── tuned_lgb.pkl
│   ├── lgb.pkl
│   ├── tuned_cb.pkl
│   ├── cb.pkl
│   └── ...
│
└── data/
    ├── combined_pareto_front.csv
    └── sample_batch.csv
```

## Local run steps

### 1. Open folder in VS Code

Open this folder:

```text
tensile_strength_scenarioB_web_app
```

### 2. Create virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run app

```bash
streamlit run app.py
```

Then open the local browser link shown in the terminal, usually:

```text
http://localhost:8501
```

## Deploy online with one link

1. Create a GitHub repository.
2. Upload the full project folder contents.
3. Go to Streamlit Community Cloud.
4. Create a new app from your GitHub repository.
5. Select `app.py` as the app entry file.
6. Deploy.

After deployment, you will get a public app link like:

```text
your-app-name.streamlit.app
```

## Notes

- If some comparison models fail to load because a dependency is missing, the app does not crash. It falls back gracefully.
- The primary RF tuned model remains the default model.
- The explanation panel is local sensitivity based. It is not formal SHAP yet, but it is visually close to an explainability module and can later be replaced with true SHAP.
- The optimizer uses `combined_pareto_front.csv` and ranks feasible mixes according to the selected goal.
