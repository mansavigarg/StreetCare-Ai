import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def render_sign_in_page():
    _, middle, _ = st.columns([1, 1.35, 1])
    with middle:
        st.html('<div style="height:5vh"></div>')
        if st.session_state.pop("session_ended", False):
            st.info(config.SESSION_ENDED_MESSAGE)
        with st.form("sign_in"):
            st.html(f'<div class="sc-signin-head">{ui.brand_block(config.DASHBOARD_TITLE, config.DASHBOARD_SUBTITLE)}'
                    '<div class="sc-welcome">Welcome back</div>'
                    '<p class="sc-lead">Sign in to continue to your staff workspace.</p></div>')
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign in", type="primary", width="stretch")
            st.html(f'<div class="sc-signin-foot">{ui.icon("shield", "primary")}<div>'
                    '<div style="color:var(--sc-ink)">Secure staff access</div>'
                    'Every review stays subject to a Highways Inspector&#39;s decision.</div></div>')
            if submitted:
                st.session_state["signed_in_user"] = call_service(
                    "POST", "/auth/signin", payload={"username": username.strip(), "password": password})
                st.rerun()
