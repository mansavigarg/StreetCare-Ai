import streamlit as st

from config import config
from dashboard.api_client import call_service


def _yes_no(value):
    return config.NOT_GIVEN_TEXT if value is None else config.BOOLEAN_YES if value else config.BOOLEAN_NO


def render_report_queue_page():
    everything = call_service("GET", "/queue", query={"defect_type": config.QUEUE_DEFECT_TYPE_ALL})
    first = config.QUEUE_FIRST_DEFECT_TYPE
    counts = [("Reports", len(everything)),
              *[(config.REPORT_STATUS_LABELS[status], sum(row["report_status"] == status for row in everything))
                for status in (config.REPORT_STATUS_AWAITING_REVIEW, config.REVIEW_STATUS_AWAITING)],
              (first, sum(first in (row["defect_type"], row["linked_defect_type"]) for row in everything)),
              *[(decision, sum(row["assessment"] == decision for row in everything)) for decision in config.DECISIONS]]
    for column, (label, value) in zip(st.columns(len(counts)), counts):
        column.metric(label, value)
    chosen_type = st.selectbox("Defect type", config.QUEUE_DEFECT_TYPE_OPTIONS)
    rows = call_service("GET", "/queue", query={"defect_type": chosen_type})
    table = [{"Position": row["queue_position"], "Reference": row["report_reference"],
              "Revision": row["revision_number"], "Defect type": row["defect_type"],
              "Linked defect type": row["linked_defect_type"], "Street": row["street_name"],
              "District": row["district"], "Channel": row["report_channel"],
              "Status": config.REPORT_STATUS_LABELS[row["report_status"]], "Assessment": row["assessment"],
              **{config.FORM_ANSWER_LABELS[column].capitalize(): row[column]
                 for column in config.REQUIREMENT_COUNT_RESULTS},
              "Live permits": row["live_permit_count"], "Conflicting permits": row["conflicting_permit_count"],
              "Triggers": row["coordination_trigger_count"], "Flags": row["disagreement_count"],
              "Documents": row["document_count"],
              "Likelihood": f"{row['repeat_likelihood']:.{config.LIKELIHOOD_DECIMALS}f} {row['likelihood_band']}",
              "Expected crew hours": f"{row['expected_crew_hours']:.{config.CREW_HOURS_DECIMALS}f}",
              "Overrun expected": _yes_no(row["overrun_expected"])} for row in rows]
    picked = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row")
    if picked.selection.rows:
        st.session_state["chosen_report"] = rows[picked.selection.rows[0]]["report_reference"]
        st.success(f"{st.session_state['chosen_report']} opens on the {config.PAGE_DETAIL} page")
