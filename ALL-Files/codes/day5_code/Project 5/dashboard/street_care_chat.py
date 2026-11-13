import streamlit as st

from config import config
from dashboard.api_client import call_service


def render_street_care_chat_page():
    turns = call_service("GET", "/chat/turns", query={"turn_limit": config.CHAT_PAGE_TURN_LIMIT})
    for turn in reversed(turns):
        st.chat_message("user").write(turn["question_text"])
        with st.chat_message("assistant"):
            st.badge(turn["question_type"], color=config.CHAT_TYPE_BADGE_COLOURS.get(turn["question_type"],
                                                                                     config.NEUTRAL_BADGE_COLOUR))
            st.write(turn["answer_text"])
            if turn["source_documents"]:
                st.caption(turn["source_documents"])
    question = st.chat_input("Ask about the street care documents", max_chars=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)
    if question:
        if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
            st.warning(f"Ask a question of at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} characters")
            return
        call_service("POST", "/chat/ask", payload={"question": question},
                     waiting_message="Searching the documents and asking the assistant",
                     failure_message=config.CHAT_FAILED_MESSAGE)
        st.rerun()
