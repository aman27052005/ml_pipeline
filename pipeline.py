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
from sklearn.cluster import DBSCAN, OPTICS

# ---------- PAGE CONFIG ----------
st.set_page_config(layout="wide", page_title="ML Pipeline Dashboard")

st.markdown(
    """
    <style>
    .stApp {background-color: #f4f8ff;}
    h1, h2, h3 {color: #1f4ed8;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🔵 Interactive ML Pipeline Dashboard")

# ---------- STEP NAV ----------
steps = [
    "1️⃣ Problem Type",
    "2️⃣ Data Input",
    "3️⃣ EDA",
    "4️⃣ Cleaning",
    "5️⃣ Feature Selection",
    "6️⃣ Split",
    "7️⃣ Model",
    "8️⃣ Training",
    "9️⃣ Metrics",
    "🔟 Tuning"
]

step = st.radio("Pipeline Steps →", steps, horizontal=True)

# ---------- GLOBAL STATE ----------
if "df" not in st.session_state:
    st.session_state.df = None

# ---------- STEP 1 ----------
if step == steps[0]:
    problem_type = st.selectbox("Select Problem Type", ["Regression", "Classification"])
    st.session_state.problem_type = problem_type

# ---------- STEP 2 ----------
elif step == steps[1]:
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.write("### Data Preview")
        st.dataframe(df.head())

        target = st.selectbox("Select Target Column", df.columns)
        st.session_state.target = target

        # PCA Visualization
        numeric_df = df.select_dtypes(include=np.number).dropna()

        pca = PCA(n_components=2)
        components = pca.fit_transform(numeric_df)

        fig = px.scatter(
            x=components[:, 0],
            y=components[:, 1],
            title="PCA Projection",
            color=numeric_df[target] if target in numeric_df else None
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- STEP 3 ----------
elif step == steps[2]:
    df = st.session_state.df

    if df is not None:
        st.write("### EDA")

        st.write("Shape:", df.shape)
        st.write("Missing Values:")
        st.write(df.isnull().sum())

        fig = px.histogram(df, x=df.columns[0])
        st.plotly_chart(fig, use_container_width=True)

        corr = df.select_dtypes(include=np.number).corr()
        fig2 = px.imshow(corr, text_auto=True, title="Correlation Matrix")
        st.plotly_chart(fig2, use_container_width=True)

# ---------- STEP 4 ----------
elif step == steps[3]:
    df = st.session_state.df

    method = st.selectbox("Fill Missing Values", ["Mean", "Median", "Mode"])

    if st.button("Apply Cleaning"):
        if method == "Mean":
            df.fillna(df.mean(), inplace=True)
        elif method == "Median":
            df.fillna(df.median(), inplace=True)
        else:
            df.fillna(df.mode().iloc[0], inplace=True)

        st.success("Missing values handled!")

    # Outliers
    if st.button("Detect Outliers"):
        iso = IsolationForest(contamination=0.05)
        outliers = iso.fit_predict(df.select_dtypes(include=np.number))

        df["Outlier"] = outliers
        st.write(df["Outlier"].value_counts())

        if st.checkbox("Remove Outliers"):
            df = df[df["Outlier"] == 1]
            st.session_state.df = df
            st.success("Outliers removed")

# ---------- STEP 5 ----------
elif step == steps[4]:
    df = st.session_state.df
    target = st.session_state.target

    X = df.drop(target, axis=1)
    y = df[target]

    method = st.selectbox("Feature Selection", ["Variance", "Correlation", "Information Gain"])

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

    st.write("Selected Features Shape:", X_new.shape)

# ---------- STEP 6 ----------
elif step == steps[5]:
    X = st.session_state.X
    y = st.session_state.y

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.success("Data Split Done")

# ---------- STEP 7 ----------
elif step == steps[6]:
    model_name = st.selectbox("Choose Model", [
        "Linear Regression",
        "SVM",
        "Random Forest",
        "KMeans"
    ])

    st.session_state.model_name = model_name

# ---------- STEP 8 ----------
elif step == steps[7]:
    k = st.slider("K-Folds", 2, 10, 3)

    X_train = st.session_state.X_train
    y_train = st.session_state.y_train

    model_name = st.session_state.model_name

    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "SVM":
        if st.session_state.problem_type == "Regression":
            model = SVR()
        else:
            model = SVC()

    elif model_name == "Random Forest":
        if st.session_state.problem_type == "Regression":
            model = RandomForestRegressor()
        else:
            model = RandomForestClassifier()

    else:
        model = KMeans(n_clusters=3)

    scores = cross_val_score(model, X_train, y_train, cv=k)

    st.write("CV Scores:", scores)
    st.write("Mean Score:", np.mean(scores))

    model.fit(X_train, y_train)
    st.session_state.model = model

# ---------- STEP 9 ----------
elif step == steps[8]:
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    preds = model.predict(X_test)

    if st.session_state.problem_type == "Regression":
        score = r2_score(y_test, preds)
        st.write("R2 Score:", score)
    else:
        score = accuracy_score(y_test, preds)
        st.write("Accuracy:", score)

# ---------- STEP 10 ----------
elif step == steps[9]:
    st.write("### Hyperparameter Tuning")

    if st.button("Run Grid Search"):
        model = RandomForestRegressor()

        params = {
            "n_estimators": [50, 100],
            "max_depth": [None, 5, 10]
        }

        grid = GridSearchCV(model, params, cv=3)
        grid.fit(st.session_state.X_train, st.session_state.y_train)

        st.write("Best Params:", grid.best_params_)
        st.write("Best Score:", grid.best_score_)