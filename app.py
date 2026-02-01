import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# =================================================
# Page Config
# =================================================
st.set_page_config(
    page_title="IPL Winner Predictor",
    page_icon="🏏",
    layout="centered"
)

# =================================================
# UI Styling (SAFE – no internal class hacks)
# =================================================
st.markdown("""
<style>

/* Background */
body {
    background-color: #0e1117;
    color: #e5e7eb;
}

/* Main padding */
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 2.5rem;
}

/* Card */
.card {
    background: #161b22;
    border-radius: 20px;
    padding: 2.5rem;
    border: 1px solid #30363d;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

/* Title */
.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #9ca3af;
    margin-bottom: 2.2rem;
}

/* Button */
.stButton>button {
    background: linear-gradient(135deg, #f59e0b, #ef4444);
    color: white;
    border-radius: 12px;
    height: 3rem;
    font-size: 1.05rem;
    font-weight: 600;
    border: none;
}

.stButton>button:hover {
    opacity: 0.9;
}

/* Result */
.result {
    background: linear-gradient(135deg, #22c55e, #16a34a);
    border-radius: 14px;
    padding: 1rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    margin-top: 1.5rem;
}

</style>
""", unsafe_allow_html=True)

# =================================================
# Load Data
# =================================================
@st.cache_data
def load_data():
    df = pd.read_csv("matches.csv")
    return df[['team1', 'team2', 'winner']].dropna()

df = load_data()

# =================================================
# Prepare Features
# =================================================
X = pd.get_dummies(df[['team1', 'team2']])
y = df['winner']

# =================================================
# Train Model
# =================================================
@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )
    model.fit(X, y)
    return model

model = train_model(X, y)

teams = sorted(set(df['team1']).union(set(df['team2'])))

# =================================================
# UI Layout
# =================================================
st.markdown('<div class="card">', unsafe_allow_html=True)

st.markdown('<div class="title">🏏 IPL Winner Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict match winners using Machine Learning</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    team1 = st.selectbox("Team 1", teams)

with col2:
    team2 = st.selectbox("Team 2", teams)

if team1 == team2:
    st.warning("Please select two different teams.")
else:
    if st.button("Predict Winner 🚀", use_container_width=True):
        input_df = pd.DataFrame([[team1, team2]], columns=['team1', 'team2'])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]

        st.markdown(
            f'<div class="result">🏆 Predicted Winner: {prediction}</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# =================================================
# Footer
# =================================================
st.markdown(
    "<p style='text-align:center;color:#6b7280;margin-top:2rem;'>Built with Streamlit • Random Forest • IPL Data</p>",
    unsafe_allow_html=True
)
