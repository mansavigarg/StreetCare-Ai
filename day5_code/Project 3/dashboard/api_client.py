import httpx
import streamlit as st

from config import config


def call_service(method, path, query=None, payload=None, waiting_message=None, failure_message=None):
    user = st.session_state.get("signed_in_user")
    headers = {config.SESSION_TOKEN_HEADER: user["session_token"]} if user else {}
    send = lambda: httpx.request(method, config.API_BASE_URL + path, params=query, json=payload, headers=headers,
                                 timeout=config.API_REQUEST_TIMEOUT_SECONDS)
    if waiting_message:
        with st.spinner(waiting_message, show_time=True):
            reply = send()
    else:
        reply = send()
    if reply.status_code == 401 and user:
        st.session_state.clear()
        st.session_state["session_ended"] = True
        st.rerun()
    if reply.status_code >= 400:
        try:
            detail = reply.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = f"{reply.status_code} {reply.text}"
        if reply.status_code == 502 and failure_message:
            detail = failure_message.format(reason=str(detail).rstrip("."))
        st.error(detail)
        st.stop()
    return reply.json()
