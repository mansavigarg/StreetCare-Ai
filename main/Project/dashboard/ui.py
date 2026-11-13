"""Presentation helpers for the dashboard: page styling, cards, pills, tables and chat bubbles.

Nothing in here calls the service or makes a decision; every page keeps its own logic and only
uses these helpers to lay the data out.
"""
from html import escape
from urllib.parse import quote

import streamlit as st

TONE_INK = {"green": "#1f6b3f", "orange": "#9a4a0b", "red": "#b42318", "violet": "#5b3cc4", "blue": "#1d5fa8",
            "yellow": "#7f5a00", "gray": "#4b5550", "primary": "#24544a", "white": "#ffffff", "muted": "#5f6b66"}
TONES = set(TONE_INK) - {"white", "muted"}

ICONS = {
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "x": '<path d="M18 6 6 18M6 6l12 12"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3M12 17h.01"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7"/>',
    "layers": '<path d="m12 2 10 5-10 5L2 7l10-5z"/><path d="m2 17 10 5 10-5M2 12l10 5 10-5"/>',
    "inbox": '<path d="M22 12h-6l-2 3h-4l-2-3H2"/><path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 '
             '2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/>',
    "pin": '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/>',
    "wrench": '<path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 '
              '6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "refer": '<path d="M7 17 17 7M7 7h10v10"/>',
    "file": '<path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z"/>'
            '<path d="M14 2v4a2 2 0 0 0 2 2h4"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
             '<path d="M12 9v4M12 17h.01"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>',
    "hash": '<path d="M4 9h16M4 15h16M10 3 8 21M16 3l-2 18"/>',
    "road": '<path d="M4 21 8 3M20 21 16 3M12 4v3M12 11v3M12 18v3"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21v-1a6 6 0 0 1 12 0v1"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="m9 12 2 2 4-4"/>',
    "map": '<path d="M9 3 3 6v15l6-3 6 3 6-3V3l-6 3-6-3zM9 3v15M15 6v15"/>',
}

STYLE = """
<style>
:root {
  --sc-bg: #f4f6f4; --sc-card: #ffffff; --sc-border: #e2e8e4; --sc-ink: #17211d; --sc-muted: #5f6b66;
  --sc-faint: #8a958f; --sc-brand: #2f6b5a; --sc-brand-dark: #24544a; --sc-brand-soft: #e7f0eb;
  --sc-shadow: 0 1px 2px rgba(23, 33, 29, .04), 0 6px 18px rgba(23, 33, 29, .05);
}
.stApp { background: var(--sc-bg); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.6rem; padding-bottom: 4rem; max-width: 1280px; }
input::-ms-reveal { display: none; }
[data-testid="stTextInput"] [data-testid="InputInstructions"] { display: none; }

/* sidebar */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid var(--sc-border); }
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] { padding-top: .75rem; }
[data-testid="stSidebarNavLink"] { border-radius: 10px; padding: .55rem .8rem; margin: 2px 0; }
[data-testid="stSidebarNavLink"] span { font-size: 1rem; color: var(--sc-ink); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--sc-brand-soft); }
[data-testid="stSidebarNavLink"][aria-current="page"] span { font-weight: 600; color: var(--sc-brand-dark); }
[data-testid="stSidebarNavSeparator"] { margin: .75rem 0; }

/* streamlit widgets */
[data-testid="stForm"] { background: var(--sc-card); border: 1px solid var(--sc-border); border-radius: 16px;
  padding: 1.6rem 1.7rem; box-shadow: var(--sc-shadow); }
[data-testid="stDataFrame"] { border: 1px solid var(--sc-border); border-radius: 14px; overflow: hidden;
  background: var(--sc-card); box-shadow: var(--sc-shadow); }
[data-testid="stExpander"] details { background: var(--sc-card); border: 1px solid var(--sc-border);
  border-radius: 14px; }
[data-baseweb="tab-list"] { gap: 1.6rem; border-bottom: 1px solid var(--sc-border); }
[data-baseweb="tab"] { padding: .6rem 0; }
[data-baseweb="tab"] p { font-size: 1.02rem; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stChatInput"] { border-radius: 14px; box-shadow: var(--sc-shadow); }
[data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--sc-ink); }

/* page header */
.sc-eyebrow { color: var(--sc-brand); font-weight: 600; font-size: .9rem; letter-spacing: .01em; margin-bottom: .2rem; }
.sc-title { font-size: 2.15rem; font-weight: 700; color: var(--sc-ink); line-height: 1.15; margin: 0 0 .45rem; }
.sc-lead { color: var(--sc-muted); font-size: 1.04rem; margin: 0; }
.sc-rule { border: 0; border-top: 1px solid var(--sc-border); margin: 1.4rem 0 1.2rem; }
.sc-section { margin: 1.6rem 0 .6rem; }
.sc-section h3 { font-size: 1.15rem; font-weight: 650; color: var(--sc-ink); margin: 0; padding: 0; }
.sc-section p { color: var(--sc-muted); margin: .2rem 0 0; font-size: .95rem; }
.sc-muted { color: var(--sc-muted); }
.sc-mono { font-family: "SFMono-Regular", Menlo, Consolas, monospace; font-size: .9em; }

/* cards */
.sc-card { background: var(--sc-card); border: 1px solid var(--sc-border); border-radius: 16px;
  box-shadow: var(--sc-shadow); padding: 1.15rem 1.35rem; }
.sc-grid { display: grid; gap: 1rem; }
.sc-kpi { background: var(--sc-card); border: 1px solid var(--sc-border); border-radius: 16px;
  box-shadow: var(--sc-shadow); padding: 1.05rem 1.3rem 1.1rem; min-width: 0; }
.sc-kpi-head { display: flex; align-items: center; gap: .6rem; color: var(--sc-muted); font-size: .97rem; }
.sc-kpi-value { font-size: 2.05rem; font-weight: 650; color: var(--sc-ink); line-height: 1.1; margin: .7rem 0 .4rem;
  overflow-wrap: anywhere; }
.sc-kpi-note { color: var(--sc-muted); font-size: .9rem; }
.sc-icon { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
  border-radius: 8px; flex: none; }
.sc-icon svg { width: 17px; height: 17px; fill: none; stroke: currentColor; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round; }
.sc-fields { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1.1rem 1.5rem; }
.sc-field-label { color: var(--sc-muted); font-size: .88rem; margin-bottom: .2rem; }
.sc-field-value { color: var(--sc-ink); font-size: 1.02rem; overflow-wrap: anywhere; }
.sc-field-wide { grid-column: span 2; }
.sc-prose { color: var(--sc-ink); font-size: 1.02rem; line-height: 1.65; white-space: pre-wrap; margin: 0; }
.sc-quote { border-left: 3px solid var(--sc-brand); padding: .2rem 0 .2rem .9rem; color: var(--sc-ink);
  font-style: italic; margin-top: .5rem; }
.sc-callout { display: flex; gap: .6rem; align-items: flex-start; border-radius: 12px; padding: .75rem .95rem;
  font-size: .95rem; margin-top: .6rem; }
.sc-callout.red { background: #fdf0ef; border: 1px solid #f4d2ce; color: #8f1f15; }
.sc-callout.blue { background: #edf4fb; border: 1px solid #d3e3f3; color: #1d4f86; }
.sc-callout .sc-icon { width: 20px; height: 20px; }

/* pills */
.sc-pills { display: flex; flex-wrap: wrap; gap: .45rem; align-items: center; }
.sc-pill { display: inline-flex; align-items: center; border-radius: 7px; padding: .16rem .6rem; font-size: .86rem;
  font-weight: 500; border: 1px solid; white-space: nowrap; line-height: 1.5; }
.sc-meta { color: var(--sc-muted); font-size: .95rem; margin-left: .25rem; }
.green { background: #eaf4ee; color: #1f6b3f; border-color: #d0e6d8; }
.orange { background: #fdf2e6; color: #9a4a0b; border-color: #f4dcc0; }
.red { background: #fdeeed; color: #b42318; border-color: #f4d0cc; }
.violet { background: #f2eefc; color: #5b3cc4; border-color: #ded5f6; }
.blue { background: #eaf2fb; color: #1d5fa8; border-color: #d0e1f4; }
.yellow { background: #fcf6e0; color: #7f5a00; border-color: #efe0a6; }
.gray { background: #f1f3f2; color: #4b5550; border-color: #dfe4e1; }
.primary { background: var(--sc-brand-soft); color: var(--sc-brand-dark); border-color: #cfe1d7; }

/* html tables */
.sc-table-wrap { background: var(--sc-card); border: 1px solid var(--sc-border); border-radius: 16px;
  box-shadow: var(--sc-shadow); overflow-x: auto; }
table.sc-table { width: 100%; border-collapse: collapse; font-size: .98rem; }
table.sc-table th { text-align: left; font-weight: 500; color: var(--sc-muted); background: #f6f8f7;
  padding: .75rem 1.1rem; border-bottom: 1px solid var(--sc-border); white-space: nowrap; }
table.sc-table td { padding: .8rem 1.1rem; border-bottom: 1px solid #edf1ee; color: var(--sc-ink);
  vertical-align: top; }
table.sc-table tr:last-child td { border-bottom: 0; }
table.sc-table td + td, table.sc-table th + th { border-left: 1px solid #edf1ee; }
table.sc-table td.strong { font-weight: 600; }
table.sc-table td.dim { color: var(--sc-muted); }
.sc-sub { display: block; color: var(--sc-faint); font-size: .84rem; font-weight: 400; margin-top: .1rem; }

/* evidence */
.sc-doc { background: var(--sc-card); border: 1px solid var(--sc-border); border-radius: 16px;
  box-shadow: var(--sc-shadow); padding: 1rem 1.1rem 1.1rem; margin-bottom: 1rem; }
.sc-doc-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap; padding: .15rem .25rem .8rem; }
.sc-doc-head .sc-icon { width: 22px; height: 22px; color: var(--sc-muted); }
.sc-doc-title { font-weight: 650; color: var(--sc-ink); font-size: 1.04rem; }
.sc-doc-count { margin-left: auto; color: var(--sc-muted); font-size: .9rem; }
.sc-sentence { border: 1px solid var(--sc-border); border-left: 3px solid #cfe1d7; border-radius: 12px;
  padding: .75rem 1rem; margin-top: .6rem; background: #fcfdfc; }
.sc-sentence.flagged { border-left-color: #d9534f; }
.sc-sentence-text { color: var(--sc-ink); font-size: 1.01rem; margin: .45rem 0 .3rem; }
.sc-sentence-note { color: var(--sc-muted); font-size: .9rem; }

/* sidebar blocks */
.sc-brand { display: flex; align-items: center; gap: .7rem; }
.sc-logo { width: 40px; height: 40px; border-radius: 11px; background: var(--sc-brand); color: #fff;
  display: inline-flex; align-items: center; justify-content: center; flex: none; }
.sc-logo svg { width: 22px; height: 22px; fill: none; stroke: currentColor; stroke-width: 2.2;
  stroke-linecap: round; stroke-linejoin: round; }
.sc-brand-name { font-weight: 700; font-size: 1.25rem; color: var(--sc-ink); line-height: 1.2; }
.sc-brand-org { color: var(--sc-muted); font-size: .88rem; }
.sc-person { margin-top: 1.3rem; padding: .85rem 1rem; border-radius: 12px; background: #f6f8f7;
  border: 1px solid var(--sc-border); }
.sc-person-name { font-weight: 650; font-size: 1.05rem; color: var(--sc-ink); }
.sc-person-role { color: var(--sc-muted); font-size: .92rem; }
.sc-status { color: var(--sc-muted); font-size: .86rem; margin: .9rem .2rem .2rem; line-height: 1.6; }
.sc-status .sc-mono { display: block; padding-left: 1rem; font-size: .8rem; color: var(--sc-faint); }
.sc-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #2e9b62;
  margin-right: .45rem; }

/* sign in */
.sc-signin-head { margin-bottom: 1.6rem; }
.sc-welcome { font-size: 1.85rem; font-weight: 700; color: var(--sc-ink); margin: 1.8rem 0 .3rem; }
.sc-signin-foot { display: flex; gap: .7rem; align-items: flex-start; border-top: 1px solid var(--sc-border);
  padding-top: 1.1rem; margin-top: .9rem; color: var(--sc-muted); font-size: .92rem; }
.sc-signin-foot .sc-icon { width: 26px; height: 26px; border-radius: 50%; }

/* chat */
.sc-chat { display: flex; flex-direction: column; gap: 1rem; padding-bottom: 1rem; }
.sc-row { display: flex; gap: .7rem; align-items: flex-start; }
.sc-row.user { justify-content: flex-end; }
.sc-avatar { width: 36px; height: 36px; border-radius: 10px; display: inline-flex; align-items: center;
  justify-content: center; flex: none; }
.sc-avatar svg { width: 18px; height: 18px; fill: none; stroke: currentColor; stroke-width: 2;
  stroke-linecap: round; stroke-linejoin: round; }
.sc-avatar.bot { background: var(--sc-brand); color: #fff; }
.sc-avatar.me { background: #e3ece7; color: var(--sc-brand-dark); }
.sc-bubble { max-width: min(760px, 78%); border-radius: 16px; padding: .8rem 1.05rem; font-size: 1.01rem;
  line-height: 1.6; overflow-wrap: anywhere; }
.sc-bubble.user { background: var(--sc-brand); color: #fff; border-bottom-right-radius: 5px; white-space: pre-wrap; }
.sc-bubble.bot { background: var(--sc-card); border: 1px solid var(--sc-border); box-shadow: var(--sc-shadow);
  color: var(--sc-ink); border-top-left-radius: 5px; }
.sc-bubble.bot .sc-answer { white-space: pre-wrap; margin-top: .5rem; }
.sc-who { font-size: .8rem; color: var(--sc-faint); margin-bottom: .25rem; }
.sc-row.user .sc-who { text-align: right; }
.sc-sources { display: flex; gap: .4rem; align-items: flex-start; margin-top: .7rem; padding-top: .6rem;
  border-top: 1px dashed var(--sc-border); color: var(--sc-muted); font-size: .86rem; }
.sc-sources img { margin-top: .2rem; flex: none; }
.sc-unused { width: 14px; height: 14px; flex: none; margin-top: .2rem; fill: none; stroke: currentColor;
  stroke-width: 2; }
.sc-empty { text-align: center; padding: 2.6rem 1rem; }
.sc-empty .sc-logo { width: 52px; height: 52px; border-radius: 14px; }
.sc-empty h3 { font-size: 1.25rem; margin: 1rem 0 .3rem; color: var(--sc-ink); }

@media (max-width: 900px) {
  .sc-fields { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .sc-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
  .sc-bubble { max-width: 88%; }
}
@media (max-width: 560px) {
  .sc-grid, .sc-fields { grid-template-columns: minmax(0, 1fr) !important; }
  .sc-field-wide { grid-column: span 1; }
}
</style>
"""


def e(value):
    """Escapes any value for HTML."""
    return escape("" if value is None else str(value))


def tone(text, colours, default="gray"):
    """Maps a value to one of the pill tones, through the same colour tables the pages used for badges."""
    colour = colours.get(text, default)
    return colour if colour in TONES else default


def svg(name, colour="primary", size=18):
    """An icon as a CSS mask (st.html sanitises <svg> and data-URI images, but keeps classes and styles)."""
    return (f'<span class="sc-ic ic-{name}" style="width:{size}px;height:{size}px;color:{TONE_INK[colour]}">'
            "</span>")


def _icon_rules():
    rules = [".sc-ic { display: inline-block; flex: none; background-color: currentColor; "
             "-webkit-mask: var(--sc-mask) center / contain no-repeat; mask: var(--sc-mask) center / contain no-repeat; }"]
    for name, paths in ICONS.items():
        markup = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
                  f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>')
        rules.append(f'.ic-{name} {{ --sc-mask: url("data:image/svg+xml,{quote(markup)}"); }}')
    return "\n".join(rules)


def icon(name, colour="primary"):
    return f'<span class="sc-icon {colour}">{svg(name, colour, 17)}</span>'


def pill(text, colour="gray"):
    return f'<span class="sc-pill {colour}">{e(text)}</span>'


def pills(items, meta=None):
    """items: (text, tone) pairs. meta: optional muted text after the pills."""
    body = "".join(pill(text, colour) for text, colour in items)
    if meta:
        body += f'<span class="sc-meta">{e(meta)}</span>'
    return f'<div class="sc-pills">{body}</div>'


def show(markup):
    st.html(markup)


def inject_styles():
    st.html(STYLE.replace("</style>", _icon_rules() + "\n</style>"))


def page_header(title, lead, eyebrow="StreetCare AI"):
    show(f'<div class="sc-eyebrow">{e(eyebrow)}</div><div class="sc-title">{e(title)}</div>'
         f'<p class="sc-lead">{e(lead)}</p><hr class="sc-rule">')


def section(title, caption=None):
    show(f'<div class="sc-section"><h3>{e(title)}</h3>' + (f"<p>{e(caption)}</p>" if caption else "") + "</div>")


def kpi(label, value, icon_name, colour="primary", note=None, note_pill=None):
    """One figure card. note is muted text; note_pill is a (text, tone) pair shown instead."""
    foot = pill(*note_pill) if note_pill else f'<div class="sc-kpi-note">{e(note)}</div>' if note else ""
    return (f'<div class="sc-kpi"><div class="sc-kpi-head">{icon(icon_name, colour)}<span>{e(label)}</span></div>'
            f'<div class="sc-kpi-value">{e(value)}</div>{foot}</div>')


def kpi_grid(cards, columns):
    show(f'<div class="sc-grid" style="grid-template-columns: repeat({columns}, minmax(0, 1fr))">'
         + "".join(cards) + "</div>")


def fields_card(fields):
    """fields: (label, value, wide) triples, shown as a labelled grid inside one card."""
    cells = "".join(f'<div class="{"sc-field-wide" if wide else ""}"><div class="sc-field-label">{e(label)}</div>'
                    f'<div class="sc-field-value">{value}</div></div>' for label, value, wide in fields)
    show(f'<div class="sc-card"><div class="sc-fields">{cells}</div></div>')


def prose_card(text, heading=None):
    head = f'<div class="sc-field-label" style="margin-bottom:.5rem">{e(heading)}</div>' if heading else ""
    show(f'<div class="sc-card">{head}<p class="sc-prose">{e(str(text).strip())}</p></div>')


def callout(text, colour="red", icon_name="alert"):
    return f'<div class="sc-callout {colour}">{icon(icon_name, colour)}<div>{text}</div></div>'


def table(headers, rows):
    """rows: lists of (html, css class) cells; the html must already be escaped."""
    head = "".join(f"<th>{e(header)}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f'<td class="{css}">{cell}</td>' for cell, css in row) + "</tr>"
                   for row in rows)
    show(f'<div class="sc-table-wrap"><table class="sc-table"><thead><tr>{head}</tr></thead>'
         f"<tbody>{body}</tbody></table></div>")


def brand_block(title, organisation):
    return (f'<div class="sc-brand"><span class="sc-logo">{svg("road", "white", 22)}</span>'
            f'<div><div class="sc-brand-name">{e(title)}</div><div class="sc-brand-org">{e(organisation)}</div>'
            "</div></div>")


def user_bubble(text, who):
    return (f'<div class="sc-row user"><div style="display:flex;flex-direction:column;align-items:flex-end;'
            f'max-width:100%"><div class="sc-who">{e(who)}</div><div class="sc-bubble user">{e(text)}</div></div>'
            f'<span class="sc-avatar me">{svg("user", "primary")}</span></div>')


def answer_bubble(text, kind, colour, sources):
    source_line = (f'<div class="sc-sources">{svg("file", "muted", 14)}'
                   f"<span>{e(sources)}</span></div>") if sources else ""
    return (f'<div class="sc-row"><span class="sc-avatar bot">{svg("road", "white")}</span>'
            f'<div class="sc-bubble bot">{pill(kind, colour)}<div class="sc-answer">{e(str(text).strip())}</div>'
            f"{source_line}</div></div>")


WRITE_UP_KEYS = {"review_plan_text": "Review planner", "requirements_analysis_text": "Requirements analyst",
                 "evidence_review_text": "Evidence analyst", "prediction_analysis_text": "Risk analyst"}


def agent_writeups(review, request=None):
    """Each agent's write-up as a readable card, with the coordinator's request first when given."""
    if request:
        show(f'<div class="sc-field-label">Coordinator&#39;s request</div><div class="sc-quote">{e(request)}</div>')
    for key, title in WRITE_UP_KEYS.items():
        if review.get(key):
            show(f'<div class="sc-card" style="margin-top:.8rem"><div class="sc-field-label" '
                 f'style="margin-bottom:.4rem;font-weight:600">{e(title)}</div>'
                 f'<p class="sc-prose" style="font-size:.95rem">{e(review[key].strip())}</p></div>')


BAR_STYLE = """
<style>
.sc-bars { display: flex; flex-direction: column; gap: .15rem; }
.sc-bar-row { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(0, 3fr) 4.8rem; gap: 1rem;
  align-items: center; padding: .55rem 0; border-bottom: 1px solid #edf1ee; }
.sc-bar-row:last-child { border-bottom: 0; }
.sc-bars.drivers .sc-bar-row { grid-template-columns: minmax(0, 3fr) minmax(0, 2fr) 3rem; }
.sc-bar-label { color: var(--sc-ink); font-size: .96rem; line-height: 1.35; }
.sc-bar-track { position: relative; height: 14px; background: #f1f4f2; border-radius: 7px; }
.sc-bar-track.centred::after { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px;
  background: #b9c4be; }
.sc-bar-fill { position: absolute; top: 0; bottom: 0; border-radius: 7px; }
.sc-bar-value { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; font-size: .95rem; }
.sc-legend { display: flex; gap: 1.2rem; flex-wrap: wrap; color: var(--sc-muted); font-size: .86rem; margin-top: .7rem; }
.sc-swatch { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: .35rem; }
@media (max-width: 700px) { .sc-bar-row { grid-template-columns: minmax(0, 1fr) 4.2rem; }
  .sc-bar-track { grid-column: 1 / -1; grid-row: 2; } }
</style>
"""

BAR_COLOURS = {"up_bad": "#d4544a", "down_good": "#3f8f63", "up_more": "#d98a2b", "down_less": "#3d7fc4",
               "neutral": "#2f6b5a"}


def contribution_bars(items, number_format, up_colour, down_colour, up_text, down_text):
    """Centred bars, one per feature: to the right when it pushes the figure up, to the left when it pulls it down."""
    largest = max((abs(item["effect"]) for item in items), default=0) or 1
    rows = ""
    for item in items:
        width = abs(item["effect"]) / largest * 50
        up = item["effect"] >= 0
        position = f"left:50%;width:{width:.2f}%" if up else f"left:{50 - width:.2f}%;width:{width:.2f}%"
        colour = BAR_COLOURS[up_colour if up else down_colour]
        rows += (f'<div class="sc-bar-row"><div class="sc-bar-label">{e(item["label"])}'
                 f'<span class="sc-sub">Answer: {e(item["value"])}</span></div>'
                 f'<div class="sc-bar-track centred"><div class="sc-bar-fill" style="{position};background:{colour}">'
                 f'</div></div><div class="sc-bar-value" style="color:{colour}">'
                 f'{"+" if up else "−"}{number_format.format(abs(item["effect"]))}</div></div>')
    legend = (f'<div class="sc-legend"><span><span class="sc-swatch" style="background:{BAR_COLOURS[up_colour]}">'
              f'</span>{e(up_text)}</span><span><span class="sc-swatch" style="background:{BAR_COLOURS[down_colour]}">'
              f'</span>{e(down_text)}</span></div>')
    show(BAR_STYLE + f'<div class="sc-card"><div class="sc-bars">{rows}</div>{legend}</div>')


def driver_bars(drivers, colour="neutral"):
    """Plain bars for overall drivers: share of the total impact, with how often each pushed up or down."""
    largest = max((item["share"] for item in drivers), default=0) or 1
    rows = "".join(
        f'<div class="sc-bar-row"><div class="sc-bar-label">{e(item["label"])}'
        f'<span class="sc-sub">Pushes up in {item["pushes_up"]} · pulls down in {item["pulls_down"]}</span></div>'
        f'<div class="sc-bar-track"><div class="sc-bar-fill" style="left:0;width:{item["share"] / largest * 100:.1f}%;'
        f'background:{BAR_COLOURS[colour]}"></div></div>'
        f'<div class="sc-bar-value">{item["share"] * 100:.0f}%</div></div>' for item in drivers)
    show(BAR_STYLE + f'<div class="sc-card"><div class="sc-bars drivers">{rows}</div></div>')
