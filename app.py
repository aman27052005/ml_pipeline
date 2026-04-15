import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    layout="wide",
    page_icon="🔵"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background-color: #f0f6ff; }
h1, h2, h3 { color: #1d4ed8; }
div[data-testid="stSidebar"] { background-color: #e0ecff; }
.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🔵 ML Pipeline Dashboard")

# ---------------- SESSION INIT ----------------
for key in ["df", "target", "X", "y", "X_train", "X_test", "y_train", "y_test", "model", "problem_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------- SAFE FUNCTION ----------------
def require(key, msg):
    if st.session_state[key] is None:
        st.error(msg)
        st.stop()
    return st.session_state[key]

# ---------------- SIDEBAR ----------------
step = st.sidebar.radio("🚀 Pipeline Steps", [
    "1. Problem Type",
    "2. Upload Data",
    "3. EDA",
    "4. Cleaning",
    "5. Feature Selection",
    "6. Train-Test Split",
    "7. Model Selection",
    "8. Training",
    "9. Metrics",
    "10. Tuning"
])

# ---------------- 1. PROBLEM TYPE ----------------
if step == "1. Problem Type":
    st.header("🎯 Problem Type")
    st.session_state.problem_type = st.selectbox("Choose", ["Regression", "Classification"])

# ---------------- 2. DATA UPLOAD ----------------
elif step == "2. Upload Data":
    st.header("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.dataframe(df.head())

        st.session_state.target = st.selectbox("🎯 Select Target Column", df.columns)

        num = df.select_dtypes(include=np.number).dropna()

        if num.shape[1] > 1:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(num)

            fig = px.scatter(
                x=comp[:, 0],
                y=comp[:, 1],
                title="PCA Projection",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- 3. EDA ----------------
elif step == "3. EDA":
    df = require("df", "Upload data first")

    st.header("📊 EDA")

    st.write("Shape:", df.shape)
    st.write(df.isnull().sum())

    num = df.select_dtypes(include=np.number)

    if num.shape[1] > 0:
        fig = px.imshow(num.corr(), text_auto=True, title="Correlation")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- 4. CLEANING ----------------
elif step == "4. Cleaning":
    df = require("df", "Upload data first")

    st.header("🧹 Cleaning")

    method = st.selectbox("Missing Value Handling", ["Mean", "Median", "Mode"])

    if st.button("Apply"):
        if method == "Mean":
            df.fillna(df.mean(numeric_only=True), inplace=True)
        elif method == "Median":
            df.fillna(df.median(numeric_only=True), inplace=True)
        else:
            df.fillna(df.mode().iloc[0], inplace=True)

        st.session_state.df = df
        st.success("Missing values handled")

    if st.button("Detect Outliers"):
        num = df.select_dtypes(include=np.number)

        iso = IsolationForest(contamination=0.05, random_state=42)
        out = iso.fit_predict(num)

        df["Outlier"] = out
        st.session_state.df = df

        st.write(df["Outlier"].value_counts())

        if st.checkbox("Remove Outliers"):
            df = df[df["Outlier"] == 1]
            st.session_state.df = df
            st.success("Outliers removed")

# ---------------- 5. FEATURE SELECTION (FIXED) ----------------
elif step == "5. Feature Selection":
    df = require("df", "Upload data first")
    target = require("target", "Select target first")

    st.header("🧠 Feature Selection")

    # Remove non-numeric features
    X = df.drop(target, axis=1)
    X = X.select_dtypes(include=np.number)

    y = df[target]

    # Encode target if categorical
    if y.dtype == 'object':
        st.warning("Target is categorical → Encoding applied")
        le = LabelEncoder()
        y = le.fit_transform(y)

    method = st.selectbox("Method", ["Variance", "Correlation", "Mutual Info"])

    if method == "Variance":
        selector = VarianceThreshold(0.1)
        X_new = selector.fit_transform(X)
        X_new = pd.DataFrame(X_new)

    elif method == "Correlation":
        temp = X.copy()
        temp[target] = y
        corr = temp.corr()[target].abs()
        cols = corr[corr > 0.2].index
        cols = cols.drop(target)
        X_new = X[cols]

    else:
        if st.session_state.problem_type == "Regression":
            scores = mutual_info_regression(X, y)
        else:
            scores = mutual_info_classif(X, y)

        cols = X.columns[np.argsort(scores)[-5:]]
        X_new = X[cols]

    st.session_state.X = X_new
    st.session_state.y = y

    st.success(f"Selected Features: {X_new.shape}")

# ---------------- 6. SPLIT ----------------
elif step == "6. Train-Test Split":
    X = require("X", "Complete Feature Selection first")
    y = require("y", "Complete Feature Selection first")

    st.header("✂️ Split")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split done")

# ---------------- 7. MODEL ----------------
elif step == "7. Model Selection":
    st.header("🤖 Model")

    st.session_state.model_name = st.selectbox(
        "Choose Model",
        ["Linear Regression", "SVM", "Random Forest"]
    )

# ---------------- 8. TRAINING (FIXED) ----------------
elif step == "8. Training":
    X_train = require("X_train", "Split data first")
    y_train = require("y_train", "Split data first")

    st.header("🏋️ Training")

    model_name = st.session_state.model_name

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "SVM":
        model = SVR() if st.session_state.problem_type == "Regression" else SVC()

    else:
        model = RandomForestRegressor() if st.session_state.problem_type == "Regression" else RandomForestClassifier()

    k = st.slider("K-Fold", 2, 10, 3)

    # 🔥 FIXED LINE
    scores = cross_val_score(model, X_train, y_train, cv=k, error_score='raise')

    st.write("CV Scores:", scores)
    st.write("Mean:", np.mean(scores))

    model.fit(X_train, y_train)
    st.session_state.model = model

# ---------------- 9. METRICS ----------------
elif step == "9. Metrics":
    model = require("model", "Train model first")
    X_test = require("X_test", "Split data first")
    y_test = require("y_test", "Split data first")

    st.header("📈 Metrics")

    preds = model.predict(X_test)

    if st.session_state.problem_type == "Regression":
        st.metric("R2 Score", r2_score(y_test, preds))
    else:
        st.metric("Accuracy", accuracy_score(y_test, preds))

# ---------------- 10. TUNING ----------------
elif step == "10. Tuning":
    X_train = require("X_train", "Split data first")
    y_train = require("y_train", "Split data first")

    st.header("⚙️ Hyperparameter Tuning")

    if st.button("Run GridSearch"):
        model = RandomForestRegressor()

        params = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }

        grid = GridSearchCV(model, params, cv=3)
        grid.fit(X_train, y_train)

        st.success("Best Params Found")
        st.write(grid.best_params_)
        st.write("Best Score:", grid.best_score_)