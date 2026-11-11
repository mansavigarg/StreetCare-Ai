import streamlit as st

from config import config
from dashboard.api_client import call_service


def render_sign_in_page():
    st.title(config.DASHBOARD_TITLE)
    st.caption(config.DASHBOARD_SUBTITLE)
    if st.session_state.pop("session_ended", False):
        st.info(config.SESSION_ENDED_MESSAGE)
    with st.form("sign_in"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            st.session_state["signed_in_user"] = call_service(
                "POST", "/auth/signin", payload={"username": username.strip(), "password": password})
            st.rerun()
