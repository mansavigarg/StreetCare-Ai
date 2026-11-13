import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service

DECISION_ICONS = {config.DECISION_ORDER: "wrench", config.DECISION_INSPECT: "search",
                  config.DECISION_REFER: "refer"}


def _yes_no(value):
    return config.NOT_GIVEN_TEXT if value is None else config.BOOLEAN_YES if value else config.BOOLEAN_NO


def _share(count, total):
    return f"{round(100 * count / total) if total else 0}% of the queue"


def render_report_queue_page():
    ui.page_header(config.PAGE_QUEUE, "Every open defect report, screened against its own standard and ranked "
                                      "in working order.")
    everything = call_service("GET", "/queue", query={"defect_type": config.QUEUE_DEFECT_TYPE_ALL})
    total = len(everything)
    first = config.QUEUE_FIRST_DEFECT_TYPE
    awaiting_review = sum(row["report_status"] == config.REPORT_STATUS_AWAITING_REVIEW for row in everything)
    awaiting_decision = sum(row["report_status"] == config.REVIEW_STATUS_AWAITING for row in everything)
    first_count = sum(first in (row["defect_type"], row["linked_defect_type"]) for row in everything)
    ui.kpi_grid([ui.kpi("Open reports", total, "layers", note="In working order"),
                 ui.kpi(config.REPORT_STATUS_LABELS[config.REPORT_STATUS_AWAITING_REVIEW], awaiting_review, "inbox",
                        "gray", note="No review started yet"),
                 ui.kpi(config.REPORT_STATUS_LABELS[config.REVIEW_STATUS_AWAITING], awaiting_decision, "clock",
                        "blue", note="Reviewed, waiting for an inspector"),
                 ui.kpi(f"{first} reports", first_count, "flag", "red", note="Moved to the front of the queue")], 4)
    st.html('<div style="height:.9rem"></div>')
    ui.kpi_grid([ui.kpi(f"Assessed {decision.lower()}", sum(row["assessment"] == decision for row in everything),
                        DECISION_ICONS.get(decision, "check"), ui.tone(decision, config.DECISION_BADGE_COLOURS),
                        note=_share(sum(row["assessment"] == decision for row in everything), total))
                 for decision in config.DECISIONS], len(config.DECISIONS))
    st.html('<div style="height:1.2rem"></div>')
    chosen_type = st.segmented_control("Defect type", config.QUEUE_DEFECT_TYPE_OPTIONS,
                                       default=config.QUEUE_DEFECT_TYPE_ALL) or config.QUEUE_DEFECT_TYPE_ALL
    rows = call_service("GET", "/queue", query={"defect_type": chosen_type})
    st.caption(f"{len(rows)} reports · Select a row to open it on the {config.PAGE_DETAIL} page")
    table = [{"#": row["queue_position"], "Reference": row["report_reference"], "Defect type": row["defect_type"],
              "Linked defect": "—" if row["linked_defect_type"] == config.NO_LINKED_DEFECT
              else row["linked_defect_type"],
              "Street": row["street_name"], "District": row["district"],
              "Status": config.REPORT_STATUS_LABELS[row["report_status"]], "Assessment": row["assessment"],
              "Met": row[config.REQUIREMENTS_MET_COUNT_COLUMN],
              "Not met": row[config.REQUIREMENTS_NOT_MET_COUNT_COLUMN],
              "Not documented": row[config.REQUIREMENTS_NOT_DOCUMENTED_COUNT_COLUMN],
              "Triggers": row["coordination_trigger_count"], "Live permits": row["live_permit_count"],
              "Conflicting permits": row["conflicting_permit_count"], "Flags": row["disagreement_count"],
              "Documents": row["document_count"], "Repeat likelihood": row["repeat_likelihood"],
              "Band": row["likelihood_band"], "Crew hours": row["expected_crew_hours"],
              "Overrun": _yes_no(row["overrun_expected"]), "Channel": row["report_channel"],
              "Revision": row["revision_number"]} for row in rows]
    column_config = {
        "#": st.column_config.NumberColumn(width=50),
        "Reference": st.column_config.TextColumn(width="small"),
        "Defect type": st.column_config.TextColumn(width="medium"),
        "Linked defect": st.column_config.TextColumn(width="medium"),
        "Street": st.column_config.TextColumn(width="medium"),
        "Status": st.column_config.TextColumn(width="medium"),
        "Assessment": st.column_config.TextColumn(width=190),
        "Repeat likelihood": st.column_config.ProgressColumn(min_value=0, max_value=1,
                                                             format=f"%.{config.LIKELIHOOD_DECIMALS}f"),
        "Crew hours": st.column_config.NumberColumn(format=f"%.{config.CREW_HOURS_DECIMALS}f h"),
    }
    picked = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row",
                          column_config=column_config, height=min(38 + 35 * len(table), 640))
    if picked.selection.rows:
        st.session_state["chosen_report"] = rows[picked.selection.rows[0]]["report_reference"]
        st.success(f"{st.session_state['chosen_report']} opens on the {config.PAGE_DETAIL} page")
