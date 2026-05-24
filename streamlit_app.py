"""
IPL Score Predictor - Streamlit App (Phase 14)

Run with:
    streamlit run streamlit_app.py
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ── Load model & metadata ──────────────────────────────────
@st.cache_resource
def load_model():
    with open("output/ipl_score_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("output/feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, feature_columns

model, FEATURE_COLUMNS = load_model()

IPL_TEAMS = sorted([
    "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Delhi Capitals", "Sunrisers Hyderabad",
    "Punjab Kings", "Rajasthan Royals", "Deccan Chargers",
    "Delhi Daredevils", "Kings XI Punjab",
    "Rising Pune Supergiants", "Gujarat Lions",
    "Lucknow Super Giants", "Gujarat Titans",
])

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="IPL Score Predictor", page_icon="🏏")
st.title("🏏 IPL Score Predictor")
st.markdown("Predict the **final first-innings score** from current match state.")

col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox("Batting team", IPL_TEAMS, index=0)
with col2:
    bowling_team = st.selectbox("Bowling team",
                                [t for t in IPL_TEAMS if t != batting_team],
                                index=0)

current_score  = st.slider("Current score (runs)", 0, 200, 80)
wickets_fallen = st.slider("Wickets fallen", 0, 9, 2)
overs_done     = st.slider("Overs completed", 6.0, 20.0, 12.0, step=0.1)

if st.button("Predict Final Score", type="primary"):
    balls_done   = int(overs_done) * 6 + round((overs_done % 1) * 10)
    balls_left   = max(120 - balls_done, 0)
    wickets_left = 10 - wickets_fallen
    crr          = current_score / max(overs_done, 0.1)

    row = {col: 0 for col in FEATURE_COLUMNS}
    row.update({
        "cumulative_runs": current_score,
        "wickets_left":    wickets_left,
        "balls_left":      balls_left,
        "crr":             crr,
        "overs_done":      overs_done,
    })
    bat_col  = f"bat_{batting_team}"
    bowl_col = f"bowl_{bowling_team}"
    if bat_col  in row: row[bat_col]  = 1
    if bowl_col in row: row[bowl_col] = 1

    pred = model.predict(pd.DataFrame([row]))[0]
    final = max(int(round(pred)), current_score)

    st.success(f"### Predicted Final Score: **{final}**")
    st.metric("Current", f"{current_score}/{wickets_fallen}",
              delta=f"+{final - current_score} runs projected")
    st.caption(
        f"CRR: {crr:.2f}  |  Balls remaining: {balls_left}"
        f"  |  Wickets in hand: {wickets_left}"
    )
