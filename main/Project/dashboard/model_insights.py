import streamlit as st

from config import config
from dashboard import ui
from dashboard.api_client import call_service


def _drivers(part, colour):
    queue_column, history_column = st.columns(2, gap="large")
    with queue_column:
        ui.show('<div class="sc-field-label" style="font-weight:600;margin-bottom:.5rem">In the open queue</div>')
        ui.driver_bars(part["queue_drivers"], colour)
    with history_column:
        ui.show('<div class="sc-field-label" style="font-weight:600;margin-bottom:.5rem">In past reports</div>')
        ui.driver_bars(part["history_drivers"], colour)


def render_model_insights_page():
    ui.page_header(config.PAGE_INSIGHTS, "Why the models predict what they do: the answers that push a report "
                                         "towards being reported again, or a repair towards more crew hours.")
    insights = call_service("GET", "/insights", waiting_message="Explaining every report in the queue")
    repeat, crew, reports = insights["repeat"], insights["crew_hours"], insights["reports"]
    high = sum(row["likelihood_band"] == config.HIGH_LIKELIHOOD_BAND for row in reports)
    ui.kpi_grid([
        ui.kpi("Reports explained", len(reports), "layers", note="Every report in the open queue"),
        ui.kpi("Typical repeat likelihood", f"{repeat['base_value']:.{config.LIKELIHOOD_DECIMALS}f}", "activity",
               note=f"Across {insights['history_rows']} past reports"),
        ui.kpi("High likelihood reports", high, "flag", "red", note="Most likely to be reported again"),
        ui.kpi("Typical crew hours", f"{crew['base_value']:.{config.CREW_HOURS_DECIMALS}f}", "clock", "blue",
               note="Across past repairs")], 4)
    ui.show(ui.callout("Each bar is an answer's share of everything that moved the predictions, from SHAP values "
                       "computed on the trained models. No language model call is made.", "blue", "help"))
    ui.section("What makes a defect come back", "The answers that most move the repeat likelihood.")
    _drivers(repeat, "up_bad")
    ui.section("What makes a repair take longer", "The answers that most move the expected crew hours.")
    _drivers(crew, "up_more")
    ui.section("Why each open report may come back",
               "The answers pushing each report's repeat likelihood and crew hours up, largest first.")
    user = st.session_state["signed_in_user"]
    districts = sorted({row["district"] for row in reports})
    options = [config.QUEUE_DEFECT_TYPE_ALL, *districts]
    district = st.segmented_control("District", options,
                                    default=user["district"] if user["district"] in districts
                                    else config.QUEUE_DEFECT_TYPE_ALL) or config.QUEUE_DEFECT_TYPE_ALL
    shown = [row for row in reports if district in (config.QUEUE_DEFECT_TYPE_ALL, row["district"])]
    shown.sort(key=lambda row: row["repeat_likelihood"], reverse=True)
    st.dataframe([{"Reference": row["report_reference"], "Street": row["street_name"], "District": row["district"],
                   "Defect type": row["defect_type"], "Repeat likelihood": row["repeat_likelihood"],
                   "Band": row["likelihood_band"],
                   "Pushing a repeat up": ", ".join(row["repeat_reasons"]) or "—",
                   "Crew hours": row["expected_crew_hours"],
                   "Pushing crew hours up": ", ".join(row["crew_hours_reasons"]) or "—"} for row in shown],
                 hide_index=True, height=min(38 + 35 * len(shown), 640),
                 column_config={
                     "Repeat likelihood": st.column_config.ProgressColumn(
                         min_value=0, max_value=1, format=f"%.{config.LIKELIHOOD_DECIMALS}f"),
                     "Crew hours": st.column_config.NumberColumn(format=f"%.{config.CREW_HOURS_DECIMALS}f h"),
                     "Pushing a repeat up": st.column_config.TextColumn(width="large"),
                     "Pushing crew hours up": st.column_config.TextColumn(width="large")})
    st.caption(f"Open a report on the {config.PAGE_DETAIL} page and choose “Why this prediction” to see every "
               "answer's effect on it.")
