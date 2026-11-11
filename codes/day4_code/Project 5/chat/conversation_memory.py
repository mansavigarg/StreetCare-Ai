from config import config
from persistence.models import ChatTurn
from persistence.repository import fetch_chat_turns


def recall_conversation(session, username, question_text):
    turns = fetch_chat_turns(session, ChatTurn.username == username, config.CHAT_HISTORY_TURN_LIMIT)[::-1]
    exchanges = [f"Question: {turn['question_text']}\nAnswer: "
                 + (config.CHAT_RECORD_ANSWER_IN_HISTORY if turn["question_type"] in config.CHAT_RECORD_TYPES
                    else turn["answer_text"]) for turn in turns]
    previous = turns[-1] if turns else None
    return {"history_text": "\n\n".join(exchanges) or config.CHAT_FIRST_QUESTION_TEXT,
            "search_text": question_text if previous is None else f"{previous['question_text']} {question_text}",
            "previous_question": "" if previous is None else previous["question_text"],
            "previous_street": None if previous is None else previous["street_name"]}
