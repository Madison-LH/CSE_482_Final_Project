import io
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="NFL Play-by-Play Predictor",
    layout="wide",
)

st.title("🏈 NFL Play-by-Play Prediction App")

st.markdown(
    """
Use historical NFL play-by-play data to train simple machine learning models and
make predictions on new plays.

**How it works:**

1. Load play-by-play data from GitHub (zipped CSVs).
2. Choose a prediction task (classification or regression).
3. Select a target variable and feature columns.
4. Train a Random Forest model.
5. Use the trained model to make predictions on new data.
"""
)

# ------------------------------------------------------------
# GITHUB DATA CONFIG
# ------------------------------------------------------------
st.sidebar.header("Data Source")

GITHUB_BASE_URL = "https://github.com/Madison-LH/CSE_482_Final_Project/raw/refs/heads/main/data/"

# Map season -> zip filename on GitHub
FILE_MAP = {
    "2019": "play_by_play_2019.zip",
    "2020": "play_by_play_2020.zip",
    "2021": "play_by_play_2021.zip",
    "2023": "play_by_play_2023.zip",
}

# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_season_from_github(season: str) -> pd.DataFrame:
    """
    Load a single season of play-by-play data from a zipped CSV on GitHub.

    Assumes each ZIP contains a single CSV with the same base name.
    """
    if season not in FILE_MAP:
        raise ValueError(f"No file mapping found for season {season}")

    zip_url = GITHUB_BASE_URL + FILE_MAP[season]

    # Directly read the zipped CSV.
    df = pd.read_csv(zip_url, compression="zip", low_memory=False)
    df["season"] = int(season)
    return df


@st.cache_data(show_spinner=True)
def load_multiple_seasons(seasons):
    dfs = []
    for s in seasons:
        try:
            dfs.append(load_season_from_github(s))
        except Exception as e:
            st.error(f"Failed to load season {s}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# ------------------------------------------------------------
# SIDEBAR: SEASON SELECTION + TASK SETUP
# ------------------------------------------------------------
st.sidebar.subheader("Step 1 • Choose Season")

selected_season = st.sidebar.selectbox(
    "Season to use",
    options=list(FILE_MAP.keys()),
    index=list(FILE_MAP.keys()).index("2023") if "2023" in FILE_MAP else 0,
    help="Only one season is loaded at a time to keep the app lightweight.",
)

load_button = st.sidebar.button("Load Data")

if "data" not in st.session_state:
    st.session_state["data"] = pd.DataFrame()
    st.session_state["season"] = None

# Load data for the single selected season
if load_button or st.session_state["data"].empty or st.session_state.get("season") != selected_season:
    with st.spinner(f"Loading season {selected_season} from GitHub..."):
        st.session_state["data"] = load_season_from_github(selected_season)
        st.session_state["season"] = selected_season

df = st.session_state["data"]

if df.empty:
    st.info("No data loaded yet. Choose a season and click **Load Data** in the sidebar.")
    st.stop()

st.success(f"Loaded {len(df):,} plays from season {int(selected_season)}.")

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption("First 20 rows of the concatenated dataset.")

# ------------------------------------------------------------
# BASIC CLEANUP / COLUMN TYPING
# ------------------------------------------------------------
# We'll try to infer numeric vs non-numeric feature columns.
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

# Some sensible default feature columns if present:
default_feature_candidates = [
    "down",
    "ydstogo",
    "yardline_100",
    "qtr",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "home_timeouts_remaining",
    "away_timeouts_remaining",
]

default_features = [c for c in default_feature_candidates if c in df.columns]

suggested_classification_targets = [
    c
    for c in [
        "touchdown",
        "pass_touchdown",
        "rush_touchdown",
        "interception",
        "sack",
        "third_down_converted",
        "fourth_down_converted",
    ]
    if c in df.columns
]

suggested_regression_targets = [
    c
    for c in [
        "epa",
        "wpa",
        "home_wp",
        "away_wp",
        "td_prob",
        "fg_prob",
    ]
    if c in df.columns
]

# ------------------------------------------------------------
# MODEL SETUP: TASK TYPE, TARGET, FEATURES
# ------------------------------------------------------------
st.header("Step 2 • Define Prediction Problem")

task_type = st.radio(
    "Task type",
    ["Classification", "Regression"],
    index=0,
    horizontal=True,
)

if task_type == "Classification":
    target_options = suggested_classification_targets or numeric_cols
else:
    target_options = suggested_regression_targets or numeric_cols

if not target_options:
    st.error("No suitable target columns found. You may need to pre-process your data.")
    st.stop()

target_col = st.selectbox(
    "Target column (what are we predicting?)",
    options=target_options,
)

# Feature selection – only use numeric by default for modeling
feature_candidates = numeric_cols.copy()

# Don’t allow the target to be a feature too
if target_col in feature_candidates:
    feature_candidates.remove(target_col)

feature_cols = st.multiselect(
    "Feature columns (inputs to the model)",
    options=feature_candidates,
    default=[c for c in default_features if c in feature_candidates] or feature_candidates[:10],
    help="Only numeric columns are shown here for simplicity.",
)

if not feature_cols:
    st.warning("Select at least one feature column to continue.")
    st.stop()

st.markdown(
    f"""
**Selected target:** `{target_col}`  
**Number of features:** `{len(feature_cols)}`  
"""
)

# ------------------------------------------------------------
# TRAIN / TEST SPLIT + MODEL TRAINING
# ------------------------------------------------------------
st.header("Step 3 • Train Model")

col_left, col_right = st.columns([1, 1])

with col_left:
    test_size = st.slider(
        "Test set size (fraction)",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
    )
    random_state = st.number_input(
        "Random state (for reproducibility)",
        min_value=0,
        max_value=10_000,
        value=42,
        step=1,
    )

with col_right:
    n_estimators = st.slider(
        "Random Forest: number of trees",
        min_value=50,
        max_value=300,
        value=100,
        step=50,
    )
    max_depth = st.slider(
        "Random Forest: max depth",
        min_value=3,
        max_value=20,
        value=10,
        step=1,
    )

train_button = st.button("Train model")

if "model" not in st.session_state:
    st.session_state["model"] = None
    st.session_state["model_meta"] = {}

if train_button:
    # Drop rows with missing in target or features
    subset_cols = feature_cols + [target_col]
    data_sub = df[subset_cols].dropna()

    if data_sub.empty:
        st.error(
            "After dropping missing values in the selected features/target, no data remains."
        )
        st.stop()

    # -------------------------------
    # Limit rows to avoid memory issues
    # -------------------------------
    max_train_rows = 20_000  
    if len(data_sub) > max_train_rows:
        st.warning(
            f"Dataset has {len(data_sub):,} rows after cleaning; "
            f"randomly sampling {max_train_rows:,} rows for training to avoid crashes."
        )
        data_sub = data_sub.sample(max_train_rows, random_state=random_state)

    X = data_sub[feature_cols].values
    y = data_sub[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if (task_type == "Classification" and len(np.unique(y)) > 1) else None,
    )

    if task_type == "Classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    st.session_state["model"] = model
    st.session_state["model_meta"] = {
        "task_type": task_type,
        "target_col": target_col,
        "feature_cols": feature_cols,
    }

    st.subheader("Model Performance")

    if task_type == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.3f}")

        # Try AUC if binary classification
        unique_vals = np.unique(y_test)
        if len(unique_vals) == 2:
            try:
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_proba)
                st.write(f"**ROC AUC:** {auc:.3f}")
            except Exception:
                st.info("Could not compute ROC AUC for this target/model.")
    else:
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**R²:** {r2:.3f}")

    # Feature importance
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        fi_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        st.dataframe(fi_df, use_container_width=True)

# ------------------------------------------------------------
# PREDICTION INTERFACE
# ------------------------------------------------------------
st.header("Step 4 • Make Predictions")

if st.session_state.get("model") is None:
    st.info("Train a model first to enable predictions.")
    st.stop()

model = st.session_state["model"]
meta = st.session_state["model_meta"]
feature_cols = meta["feature_cols"]
task_type = meta["task_type"]
target_col = meta["target_col"]

mode = st.radio(
    "Prediction mode",
    ["Single play (form)", "Batch (upload CSV)"],
    horizontal=True,
)

if mode == "Single play (form)":
    st.subheader("Single Play Prediction")

    # We’ll use the training data ranges to give reasonable defaults
    data_sub = df[feature_cols].dropna()
    input_values = {}

    cols_per_row = 3
    rows = [
        feature_cols[i : i + cols_per_row]
        for i in range(0, len(feature_cols), cols_per_row)
    ]

    for row in rows:
        c_row = st.columns(len(row))
        for col_name, col in zip(row, c_row):
            with col:
                col_data = data_sub[col_name]
                v_min = float(col_data.min())
                v_max = float(col_data.max())
                v_mean = float(col_data.mean())
                input_values[col_name] = st.number_input(
                    col_name,
                    value=round(v_mean, 2),
                    min_value=v_min,
                    max_value=v_max,
                )

    if st.button("Predict for this play"):
        X_new = np.array([[input_values[c] for c in feature_cols]])
        pred = model.predict(X_new)[0]

        if task_type == "Classification":
            st.write(f"**Predicted {target_col}:** {pred}")
            # Human-friendly explanation
            if pred == 1 or pred == 1.0:
                st.info(f"A predicted value of **1** means the event `{target_col}` is expected to occur.")
            else:
                st.info(f"A predicted value of **0** means the event `{target_col}` is predicted **not** to occur.")

            # Try to show probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_new)[0]
                st.write("Class probabilities:", dict(zip(model.classes_, proba.round(3))))
        else:
            st.write(f"**Predicted {target_col}:** {pred:.4f}")

else:
    st.subheader("Batch Prediction from CSV")

    st.markdown(
        textwrap.dedent(
            f"""
            Upload a CSV file containing at least the following columns:

            ```text
            {", ".join(feature_cols)}
            ```

            The app will compute predictions and let you download the results.
            """
        )
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        missing = [c for c in feature_cols if c not in new_df.columns]
        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
            st.stop()

        X_new = new_df[feature_cols].values
        preds = model.predict(X_new)

        result_df = new_df.copy()
        result_df[f"pred_{target_col}"] = preds

        st.success(f"Generated predictions for {len(result_df):,} rows.")
        st.dataframe(result_df.head(20), use_container_width=True)

        # Provide download
        csv_buffer = io.StringIO()
        result_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name="nfl_predictions.csv",
            mime="text/csv",
        )

