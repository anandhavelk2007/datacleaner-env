import streamlit_app as st
import pandas as pd
from env.environment import DataCleanerEnv
from env.models import Action

st.set_page_config(page_title="DataCleaner Demo", layout="wide")
st.title("🧹 DataCleaner Environment Demo")
st.markdown("An interactive UI for the DataCleaner OpenEnv environment.")

task = st.selectbox("Choose Task", ["easy", "medium", "hard"])
env = DataCleanerEnv()
obs = env.reset(task)

st.subheader("Observation")
st.write("**Description:**", obs.description)
st.write("**Dataset Summary:**")
st.json(obs.dataset_summary)
st.write("**Column Issues:**", obs.column_issues)

st.subheader("Action")
col1, col2, col3 = st.columns(3)
with col1:
    action_type = st.selectbox("Action Type", ["impute", "fix_date", "normalize_cat", "skip"])
with col2:
    column = st.text_input("Column Name")
with col3:
    if action_type == "impute":
        method = st.selectbox("Method", ["mean", "median", "mode"])
    elif action_type == "normalize_cat":
        target = st.text_input("Target Value", "USA")
    else:
        method = None
        target = None

if st.button("Apply Action"):
    action = Action(type=action_type, column=column, method=method, target=target)
    obs, reward, done, info = env.step(action)
    st.success(f"Reward: {reward.value:.3f} - {reward.reason}")
    st.info(f"Current Score: {info['score']:.3f}")
    st.write("Updated Data:")
    st.dataframe(env.state.data)
    if done:
        st.balloons()
        st.success("Task completed!")