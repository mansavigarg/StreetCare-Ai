import httpx
import streamlit as st

from config import config
from dashboard.api_client import call_service
from dashboard.audit_trail import render_audit_trail_page
from dashboard.decision_queue import render_decision_queue_page
from dashboard.report_detail import render_report_detail_page
from dashboard.report_queue import render_report_queue_page
from dashboard.sign_in import render_sign_in_page
from dashboard.street_care_chat import render_street_care_chat_page

PAGE_RENDERERS = {config.PAGE_QUEUE: render_report_queue_page, config.PAGE_DETAIL: render_report_detail_page,
                  config.PAGE_DECISION: render_decision_queue_page, config.PAGE_AUDIT: render_audit_trail_page,
                  config.PAGE_CHAT: render_street_care_chat_page}


def main():
    st.set_page_config(page_title=config.DASHBOARD_TITLE, page_icon=config.DASHBOARD_ICON, layout="wide")
    st.markdown(config.DASHBOARD_PAGE_STYLE, unsafe_allow_html=True)
    try:
        httpx.get(config.API_BASE_URL + "/docs", timeout=config.API_REQUEST_TIMEOUT_SECONDS)
    except httpx.HTTPError:
        st.error(f"The service is not answering at {config.API_BASE_URL}. Start it from the Project folder with: "
                 f"python run_api_service.py")
        return
    user = st.session_state.get("signed_in_user")
    if user is None:
        st.navigation([st.Page(render_sign_in_page, title=config.DASHBOARD_TITLE, icon=config.DASHBOARD_ICON)]).run()
        return
    st.sidebar.write(f"**{user['display_name']}**")
    st.sidebar.caption(", ".join(part for part in (user["role"], user["district"]) if part))
    if st.sidebar.button("Sign out"):
        call_service("POST", "/auth/signout")
        st.session_state.clear()
        st.rerun()
    st.navigation([st.Page(PAGE_RENDERERS[page], title=page, icon=config.PAGE_ICONS[page],
                           url_path=config.PAGE_URL_PATHS[page]) for page in user["pages"]]).run()


if __name__ == "__main__":
    main()
