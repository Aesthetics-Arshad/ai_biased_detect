import subprocess

subprocess.run(["pip", "install", "plotly"])
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 🎨 Set Streamlit Page Config - Futuristic Dashboard Mode
st.set_page_config(page_title="AI Bias Detection", layout="wide")

# 🏆 Premium Dark Theme Styling
st.markdown(
    """
    <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        .title { font-size: 45px; font-weight: bold; color: #ff4b4b; text-align: center; }
        .subtitle { font-size: 22px; color: #bbb; text-align: center; }
        .stButton>button { background-color: #ff4b4b; color: white; font-size: 18px; }
        .glass { 
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""",
    unsafe_allow_html=True,
)

# 🏆 Title & Subtitle
st.markdown("<h1 class='title'>🚀 AI Bias Detection Model</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>🔍 Discover how AI decisions are influenced by bias.</p>",
    unsafe_allow_html=True,
)


# 📌 Create AI Dataset with Bias Simulation
@st.cache_data
def create_data(biased=True):
    np.random.seed(42)
    size = 3000

    data = {
        "Experience": np.random.randint(1, 21, size),
        "Gender": np.random.choice(["Male", "Female"], size, p=[0.6, 0.4]),
        "Age": np.random.randint(18, 60, size),
        "Ethnicity": np.random.choice(["White", "Black", "Asian", "Hispanic"], size),
        "Education": np.random.choice(
            ["High School", "Bachelors", "Masters", "PhD"], size
        ),
        "Company_Size": np.random.choice(["Startup", "Medium", "Enterprise"], size),
        "Hired": np.random.choice([0, 1], size, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})
    df["Ethnicity"] = df["Ethnicity"].map(
        {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3}
    )
    df["Education"] = df["Education"].map(
        {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}
    )
    df["Company_Size"] = df["Company_Size"].map(
        {"Startup": 0, "Medium": 1, "Enterprise": 2}
    )

    if biased:
        df.loc[(df["Gender"] == 1), "Hired"] = np.random.choice(
            [0, 0, 1], len(df[df["Gender"] == 1]), p=[0.75, 0.2, 0.05]
        )
        df.loc[(df["Ethnicity"] == 1), "Hired"] = np.random.choice(
            [0, 0, 1], len(df[df["Ethnicity"] == 1]), p=[0.8, 0.1, 0.1]
        )

    return df


# 📌 Train AI Bias Model Using RandomForest
@st.cache_data
def train_model(df):
    X = df[["Experience", "Gender", "Age", "Ethnicity", "Education", "Company_Size"]]
    y = df["Hired"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy


# 📊 Dataset Selection
dataset_type = st.radio(
    "📊 Choose Dataset Type:", ["Biased", "Unbiased"], horizontal=True
)
df = create_data(biased=(dataset_type == "Biased"))
model, accuracy = train_model(df)

# 🎯 AI Performance Metrics
st.subheader("📈 AI Model Performance")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🎯 Model Accuracy", f"{accuracy:.2%}")
with col2:
    st.metric("📊 Total Applicants", f"{len(df)}")
with col3:
    st.metric(
        "🚀 Bias Simulation", "Active" if dataset_type == "Biased" else "Inactive"
    )

# 🔥 AI Bias Heatmap
st.subheader("📊 Bias Heatmap")
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(
    pd.crosstab(df["Ethnicity"], df["Hired"]), cmap="coolwarm", annot=True, fmt="d"
)
plt.xlabel("Hired (1=Yes, 0=No)")
plt.ylabel("Ethnicity")
st.pyplot(fig)

# 📌 Gender-Based Hiring Bias
st.subheader("📊 Gender-Based Hiring Rates")
gender_fig = px.histogram(
    df, x="Gender", color="Hired", barmode="group", title="Hiring Bias Across Genders"
)
st.plotly_chart(gender_fig, use_container_width=True)

# 🎯 AI Prediction Simulation
st.subheader("🔍 Try AI Bias Yourself")
experience = st.slider("Experience (Years)", 1, 20, 5)
gender = st.radio("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 60, 30)
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic"])
education = st.selectbox("Education", ["High School", "Bachelors", "Masters", "PhD"])
company_size = st.selectbox("Company Size", ["Startup", "Medium", "Enterprise"])

gender = 0 if gender == "Male" else 1
ethnicity = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3}[ethnicity]
education = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}[education]
company_size = {"Startup": 0, "Medium": 1, "Enterprise": 2}[company_size]

input_data = pd.DataFrame(
    [[experience, gender, age, ethnicity, education, company_size]],
    columns=["Experience", "Gender", "Age", "Ethnicity", "Education", "Company_Size"],
)

prediction = model.predict(input_data)
prediction_result = "✅ Hired!" if prediction[0] == 1 else "❌ Not Hired!"

st.subheader("📢 AI Hiring Decision")
st.markdown(
    f"<h2 style='text-align: center; color: {'#00cc66' if prediction[0] == 1 else '#ff4b4b'};'>{prediction_result}</h2>",
    unsafe_allow_html=True,
)
