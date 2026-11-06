"""Visual identity of the dashboard: pastel palette, CSS and small HTML components."""
import html

# ---------------------------------------------------------------
# Palette (one colour per method, used on every page and chart)
# ---------------------------------------------------------------
TEXT = "#2F3142"
MUTED = "#6B6F85"
ACCENT = "#8B7FD9"          # lavender: headings, highlights
BORDER = "#ECE8F6"
GRID = "#EEEBF5"
BG_SOFT = "#F6F4FC"

METHOD_COLORS = {
    "Item-Item CF": "#7FB3E0",            # sky blue
    "Common Neighbors": "#F4A88C",        # peach
    "Personalized PageRank": "#7CCBA2",   # mint
    "Popularity (reference)": "#BDB3DA",  # lavender grey
}
METHOD_SOFT = {
    "Item-Item CF": "#E6F1FB",
    "Common Neighbors": "#FDEDE6",
    "Personalized PageRank": "#E3F5EB",
    "Popularity (reference)": "#F0EDF8",
}
SHORT_NAMES = {
    "Item-Item CF": "Item-Item CF",
    "Common Neighbors": "Common Neighbors",
    "Personalized PageRank": "PageRank (PPR)",
    "Popularity (reference)": "Popularity",
}
EVENT_COLORS = {"view": "#A9C9EC", "add to cart": "#F7C59F", "transaction": "#8FD3B0"}
NODE_COLORS = {"center": "#9B8CE6", "user": "#A9C9EC", "product": "#F7C59F", "recommended": "#6CC497"}
CARD_ACCENTS = ["#B9A9F2", "#9CC5E8", "#F7B7A3", "#A8D8B9", "#F5D58C", "#E8B4D8"]

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1300px; }}
h1, h2, h3 {{ color: {TEXT}; letter-spacing: -0.01em; }}
section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}

.hero {{
  background: linear-gradient(120deg, #EEE9FD 0%, #E5F1FB 50%, #E6F6EE 100%);
  border: 1px solid {BORDER}; border-radius: 20px; padding: 26px 30px; margin-bottom: 18px;
}}
.hero h1 {{ font-size: 1.9rem; margin: 0 0 6px 0; padding: 0; }}
.hero p {{ color: {MUTED}; margin: 0; font-size: 1rem; }}

.page-title {{ margin: 0 0 2px 0; font-size: 1.75rem; font-weight: 700; color: {TEXT}; }}
.page-sub {{ color: {MUTED}; margin: 0 0 18px 0; font-size: 0.98rem; }}
.section {{ font-size: 1.12rem; font-weight: 650; color: {TEXT}; margin: 22px 0 8px 0;
            padding-left: 10px; border-left: 4px solid {ACCENT}; }}

.kpi {{ background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 16px; padding: 14px 16px;
        box-shadow: 0 1px 3px rgba(70, 60, 140, 0.06); height: 100%; border-top: 4px solid var(--accent); }}
.kpi .label {{ font-size: 0.74rem; color: {MUTED}; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi .value {{ font-size: 1.55rem; font-weight: 700; color: {TEXT}; line-height: 1.25; margin-top: 2px; }}
.kpi .sub {{ font-size: 0.8rem; color: {MUTED}; margin-top: 2px; }}

.note {{ background: {BG_SOFT}; border: 1px solid {BORDER}; border-radius: 14px; padding: 12px 16px;
         color: {TEXT}; font-size: 0.93rem; margin: 6px 0 10px 0; }}
.pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600;
         color: {TEXT}; margin-right: 6px; }}

.rec-card {{ border-radius: 16px; padding: 14px 16px; border: 1px solid {BORDER}; height: 100%; }}
.rec-card .title {{ font-weight: 700; font-size: 0.98rem; color: {TEXT}; margin-bottom: 2px; }}
.rec-card .meta {{ font-size: 0.78rem; color: {MUTED}; margin-bottom: 8px; }}
.rec-card ol {{ margin: 0; padding-left: 1.2rem; }}
.rec-card li {{ font-size: 0.92rem; color: {TEXT}; margin: 2px 0; font-variant-numeric: tabular-nums; }}
.hit {{ color: #2E8B57; font-weight: 700; }}

.steps {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 4px 0 8px 0; }}
.step {{ flex: 1 1 150px; background: #FFFFFF; border: 1px solid {BORDER}; border-radius: 14px; padding: 10px 12px; }}
.step b {{ display: block; color: {TEXT}; font-size: 0.9rem; }}
.step span {{ color: {MUTED}; font-size: 0.8rem; }}
.step .n {{ display: inline-block; width: 22px; height: 22px; border-radius: 50%; background: #EEE9FD;
            color: {ACCENT}; text-align: center; font-weight: 700; font-size: 0.8rem; line-height: 22px;
            margin-bottom: 4px; }}
</style>
"""


def esc(x):
    return html.escape(str(x))


def kpi(label, value, sub="", accent=None):
    accent = accent or CARD_ACCENTS[0]
    return (f'<div class="kpi" style="--accent:{accent}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{esc(value)}</div><div class="sub">{esc(sub)}</div></div>')


def pill(text, color):
    return f'<span class="pill" style="background:{color}">{esc(text)}</span>'


def fmt_int(x):
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "–"


def fmt_num(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return "–"
