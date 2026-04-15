import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ML Pipeline Dashboard",
    layout="wide",
    page_icon="🔵"
)

# ---------------- MODERN UI ----------------
st.markdown("""
<style>
.stApp {
    background-color: #f0f6ff;
}

h1, h2, h3 {
    color: #1d4ed8;
}

div[data-testid="stSidebar"] {
    background-color: #e0ecff;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🔵 ML Pipeline Dashboard")

# ---------------- SIDEBAR NAV ----------------
step = st.sidebar.radio("🚀 Pipeline Steps", [
    "Problem Type",
    "Data Upload",
    "EDA",
    "Cleaning",
    "Feature Selection",
    "Train-Test Split",
    "Model Selection",
    "Training",
    "Metrics",
    "Hyperparameter Tuning"
])

# ---------------- SESSION ----------------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------------- 1. PROBLEM TYPE ----------------
if step == "Problem Type":
    st.header("🎯 Select Problem Type")
    st.session_state.problem_type = st.selectbox(
        "Choose Type", ["Regression", "Classification"]
    )

# ---------------- 2. DATA UPLOAD ----------------
elif step == "Data Upload":
    st.header("📂 Upload Dataset")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.dataframe(df.head())

        target = st.selectbox("🎯 Select Target Column", df.columns)
        st.session_state.target = target

        # PCA
        numeric_df = df.select_dtypes(include=np.number).dropna()

        if len(numeric_df.columns) > 1:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(numeric_df)

            fig = px.scatter(
                x=comp[:, 0],
                y=comp[:, 1],
                title="PCA Projection",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------------- 3. EDA ----------------
elif step == "EDA":
    df = st.session_state.df

    if df is not None:
        st.header("📊 Exploratory Data Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write("Shape:", df.shape)
            st.write("Missing Values:")
            st.write(df.isnull().sum())

        with col2:
            fig = px.histogram(df, x=df.columns[0], template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        corr = df.select_dtypes(include=np.number).corr()

        fig2 = px.imshow(corr, text_auto=True, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- 4. CLEANING ----------------
elif step == "Cleaning":
    df = st.session_state.df

    if df is not None:
        st.header("🧹 Data Cleaning")

        method = st.selectbox("Fill Missing Values", ["Mean", "Median", "Mode"])

        if st.button("Apply Cleaning"):
            if method == "Mean":
                df.fillna(df.mean(), inplace=True)
            elif method == "Median":
                df.fillna(df.median(), inplace=True)
            else:
                df.fillna(df.mode().iloc[0], inplace=True)

            st.success("Missing values handled")

        if st.button("Detect Outliers"):
            iso = IsolationForest(contamination=0.05)
            outliers = iso.fit_predict(df.select_dtypes(include=np.number))

            df["Outlier"] = outliers

            fig = px.histogram(df, x="Outlier", template="plotly_white")
            st.plotly_chart(fig)

            if st.checkbox("Remove Outliers"):
                df = df[df["Outlier"] == 1]
                st.session_state.df = df
                st.success("Outliers removed")

# ---------------- 5. FEATURE SELECTION ----------------
elif step == "Feature Selection":
    df = st.session_state.df
    target = st.session_state.target

    if df is not None:
        st.header("🧠 Feature Selection")

        X = df.drop(target, axis=1)
        y = df[target]

        method = st.selectbox("Method", ["Variance", "Correlation", "Mutual Info"])

        if method == "Variance":
            selector = VarianceThreshold(0.1)
            X_new = selector.fit_transform(X)

        elif method == "Correlation":
            corr = df.corr()[target].abs()
            cols = corr[corr > 0.2].index
            X_new = df[cols].drop(target, axis=1)

        else:
            if st.session_state.problem_type == "Regression":
                scores = mutual_info_regression(X, y)
            else:
                scores = mutual_info_classif(X, y)

            cols = X.columns[np.argsort(scores)[-5:]]
            X_new = X[cols]

        st.session_state.X = X_new
        st.session_state.y = y

        st.success(f"Selected Shape: {X_new.shape}")

# ---------------- 6. SPLIT ----------------
elif step == "Train-Test Split":
    X = st.session_state.X
    y = st.session_state.y

    st.header("✂️ Data Split")

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Split completed")

# ---------------- 7. MODEL ----------------
elif step == "Model Selection":
    st.header("🤖 Choose Model")

    st.session_state.model_name = st.selectbox(
        "Model",
        ["Linear Regression", "SVM", "Random Forest", "KMeans"]
    )

# ---------------- 8. TRAINING ----------------
elif step == "Training":
    st.header("🏋️ Model Training")

    k = st.slider("K-Folds", 2, 10, 3)

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train
    model_name = st.session_state.model_name

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "SVM":
        model = SVR() if st.session_state.problem_type == "Regression" else SVC()

    elif model_name == "Random Forest":
        model = RandomForestRegressor() if st.session_state.problem_type == "Regression" else RandomForestClassifier()

    else:
        model = KMeans(n_clusters=3)

    scores = cross_val_score(model, X_train, y_train, cv=k)

    st.write("CV Scores:", scores)
    st.write("Mean:", np.mean(scores))

    model.fit(X_train, y_train)
    st.session_state.model = model

# ---------------- 9. METRICS ----------------
elif step == "Metrics":
    st.header("📈 Performance")

    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    preds = model.predict(X_test)

    if st.session_state.problem_type == "Regression":
        score = r2_score(y_test, preds)
        st.metric("R2 Score", round(score, 3))
    else:
        score = accuracy_score(y_test, preds)
        st.metric("Accuracy", round(score, 3))

# ---------------- 10. TUNING ----------------
elif step == "Hyperparameter Tuning":
    st.header("⚙️ Hyperparameter Tuning")

    if st.button("Run GridSearch"):
        model = RandomForestRegressor()

        params = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }

        grid = GridSearchCV(model, params, cv=3)
        grid.fit(st.session_state.X_train, st.session_state.y_train)

        st.success("Best Parameters Found!")
        st.write(grid.best_params_)
        st.write("Score:", grid.best_score_)