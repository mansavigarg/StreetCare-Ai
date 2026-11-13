import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def render_street_care_chat_page():
    ui.page_header(config.PAGE_CHAT, "Ask about a defect standard, the street care rules, a statement guide or a "
                                     "report. Answers come only from the council's documents and records.")
    user = st.session_state["signed_in_user"]
    turns = call_service("GET", "/chat/turns", query={"turn_limit": config.CHAT_PAGE_TURN_LIMIT})
    if not turns:
        ui.show('<div class="sc-card sc-empty">'
                f'<span class="sc-logo">{ui.svg("road", "white", 26)}</span>'
                '<h3>Start a conversation</h3><p class="sc-lead">Try “What does the footway standard require?” '
                'or “What is the status of SC0052109?”</p></div>')
    bubbles = []
    for turn in reversed(turns):
        bubbles.append(ui.user_bubble(turn["question_text"], user["display_name"]))
        bubbles.append(ui.answer_bubble(turn["answer_text"], turn["question_type"],
                                        ui.tone(turn["question_type"], config.CHAT_TYPE_BADGE_COLOURS),
                                        turn["source_documents"]))
    if bubbles:
        ui.show('<div class="sc-chat">' + "".join(bubbles) + "</div>")
    question = st.chat_input("Ask about a standard, a rule, a statement guide or a report",
                             max_chars=config.CHAT_QUESTION_MAXIMUM_CHARACTERS)
    if question:
        if len(question.strip()) < config.CHAT_QUESTION_MINIMUM_CHARACTERS:
            st.warning(f"Ask a question of at least {config.CHAT_QUESTION_MINIMUM_CHARACTERS} characters")
            return
        ui.show('<div class="sc-chat" style="margin-top:1rem">' + ui.user_bubble(question, user["display_name"])
                + "</div>")
        call_service("POST", "/chat/ask", payload={"question": question},
                     waiting_message="Searching the documents and asking the assistant",
                     failure_message=config.CHAT_FAILED_MESSAGE)
        st.rerun()
