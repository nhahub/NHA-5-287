"""Look and feel: soft off-white and dark-slate palettes, one muted accent, system fonts (no external requests), no gradients.
Both palettes live in nexusml/palette.py; `python -m nexusml.ui.theme` writes the matching Streamlit config files."""
import os
import sys
from pathlib import Path

from .. import palette
from ._st import st

APP_NAME = os.environ.get("NEXUSML_APP_NAME", "NexusML")
FONT = '"Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'


def css(mode: str = "") -> str:
    t = palette.PALETTES[mode or palette.mode()]
    on_accent = "#FFFFFF" if (mode or palette.mode()) == "light" else "#0E1514"
    return f"""
:root {{ --bg:{t['bg']}; --surface:{t['surface']}; --sidebar:{t['sidebar']}; --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']}; --faint:{t['faint']};
  --accent:{t['accent']}; --accent-hover:{t['accent_hover']}; --accent-soft:{t['accent_soft']}; --positive:{t['positive']}; --caution:{t['caution']}; --negative:{t['negative']};
  --positive-soft:{t['positive_soft']}; --caution-soft:{t['caution_soft']}; --negative-soft:{t['negative_soft']}; }}
html, body, .stApp, button, input, textarea, p, li, label, h1, h2, h3, h4, [data-testid="stMarkdownContainer"] {{ font-family:{FONT}; }}
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {{ font-family:"Material Symbols Rounded" !important; }}
.stApp {{ background:var(--bg); color:var(--text); }}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{ visibility:hidden; height:0; }}
[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ max-width:1180px; padding-top:2.4rem; padding-bottom:4rem; }}
h1, h2, h3, h4 {{ color:var(--text); letter-spacing:-0.012em; font-weight:600; }}
h1 {{ font-size:1.65rem; margin-bottom:.15rem; }} h2 {{ font-size:1.25rem; }} h3 {{ font-size:1.05rem; }}
p, li, label, .stMarkdown {{ color:var(--text); line-height:1.55; }}
.stCaption, [data-testid="stCaptionContainer"] {{ color:var(--muted); }}
a {{ color:var(--accent); }}

/* sidebar */
[data-testid="stSidebar"] {{ background:var(--sidebar); border-right:1px solid var(--border); }}
[data-testid="stSidebarNav"] {{ padding-top:.25rem; }}
[data-testid="stSidebarNavLink"] {{ border-radius:8px; padding:.38rem .65rem; color:var(--muted); font-weight:500; }}
[data-testid="stSidebarNavLink"]:hover {{ background:var(--accent-soft); color:var(--text); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{ background:var(--accent-soft); color:var(--accent); font-weight:600; }}
[data-testid="stSidebarNavSeparator"] {{ color:var(--faint); }}

/* cards, panels */
[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; box-shadow:0 1px 2px rgba(0,0,0,.04); }}
[data-testid="stMetric"] {{ background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:.85rem 1rem; }}
[data-testid="stMetricLabel"] {{ color:var(--muted); }}
[data-testid="stExpander"] {{ border:1px solid var(--border); border-radius:10px; background:var(--surface); }}
[data-testid="stExpander"] summary {{ font-weight:500; }}
[data-testid="stDataFrame"] {{ border:1px solid var(--border); border-radius:10px; overflow:hidden; }}
[data-testid="stAlert"] {{ border-radius:10px; }}

/* controls */
.stButton > button, [data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button {{ background:var(--surface); color:var(--text); border:1px solid var(--border); border-radius:9px; font-weight:500; padding:.4rem .95rem; }}
.stButton > button:hover, [data-testid="stBaseButton-secondary"]:hover {{ border-color:var(--accent); color:var(--accent); background:var(--surface); }}
[data-testid="stBaseButton-primary"], .stButton > button[kind="primary"] {{ background:var(--accent); color:{on_accent}; border:1px solid var(--accent); border-radius:9px; font-weight:600; }}
[data-testid="stBaseButton-primary"]:hover {{ background:var(--accent-hover); border-color:var(--accent-hover); color:{on_accent}; }}
[data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] > div {{ background:var(--surface) !important; border-color:var(--border) !important; border-radius:9px !important; }}
[data-baseweb="tab-list"] {{ gap:.25rem; border-bottom:1px solid var(--border); }}
[data-baseweb="tab"] {{ color:var(--muted); font-weight:500; padding:.5rem .9rem; }}
[data-baseweb="tab"][aria-selected="true"] {{ color:var(--accent); }}
[data-baseweb="tab-highlight"] {{ background:var(--accent); height:2px; }}

/* conversation */
[data-testid="stChatMessage"] {{ background:transparent; padding:.55rem 0; }}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background:var(--accent-soft); border-radius:12px; padding:.7rem 1rem; }}
[data-testid="stChatInput"] {{ background:var(--surface); border:1px solid var(--border); border-radius:14px; }}
[data-testid="stChatInput"]:focus-within {{ border-color:var(--accent); }}

/* page furniture */
.nx-brand {{ display:flex; align-items:center; gap:.6rem; padding:.2rem .2rem .9rem; }}
.nx-mark {{ width:28px; height:28px; border-radius:8px; background:var(--accent); display:inline-flex; align-items:center; justify-content:center; color:{on_accent}; font-weight:700; font-size:.95rem; }}
.nx-name {{ font-weight:650; font-size:1.05rem; letter-spacing:-0.01em; color:var(--text); }}
.nx-sub {{ color:var(--muted); margin:0 0 1.2rem; font-size:.98rem; }}
.nx-eyebrow {{ color:var(--faint); text-transform:uppercase; letter-spacing:.07em; font-size:.72rem; font-weight:600; margin:.2rem 0 .3rem; }}
.nx-muted {{ color:var(--muted); }}
.nx-badge {{ display:inline-block; padding:.12rem .6rem; border-radius:999px; font-size:.76rem; font-weight:600; border:1px solid transparent; }}
.nx-ok {{ background:var(--positive-soft); color:var(--positive); }} .nx-warn {{ background:var(--caution-soft); color:var(--caution); }}
.nx-bad {{ background:var(--negative-soft); color:var(--negative); }} .nx-info {{ background:var(--accent-soft); color:var(--accent); }} .nx-neutral {{ background:var(--sidebar); color:var(--muted); border-color:var(--border); }}
.nx-row {{ display:flex; justify-content:space-between; align-items:center; gap:1rem; }}
.nx-empty {{ text-align:center; padding:2.2rem 1rem; color:var(--muted); }}
.nx-empty h3 {{ color:var(--text); margin-bottom:.2rem; }}
.nx-step {{ color:var(--muted); font-size:.9rem; }}
"""


def apply() -> None:
    st.markdown(f"<style>{css()}</style>", unsafe_allow_html=True)


def brand_block() -> None:
    mark = (APP_NAME[:1] or "M").upper()
    st.markdown(f'<div class="nx-brand"><span class="nx-mark">{mark}</span><span class="nx-name">{APP_NAME}</span></div>', unsafe_allow_html=True)


# Streamlit config files
def config_toml(mode: str = "light") -> str:
    t = palette.PALETTES[mode]
    return f"""# Generated by `python -m nexusml.ui.theme`. Edit palette.py, not this file.
[theme]
base = "{mode}"
primaryColor = "{t['accent']}"
backgroundColor = "{t['bg']}"
secondaryBackgroundColor = "{t['sidebar']}"
textColor = "{t['text']}"
font = "sans serif"

[client]
toolbarMode = "minimal"

[browser]
gatherUsageStats = false

[server]
headless = true
enableXsrfProtection = true
maxUploadSize = 200
fileWatcherType = "none"
"""


def write_configs(root: Path) -> None:
    d = Path(root) / ".streamlit"
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.toml").write_text(config_toml("light"), encoding="utf-8")
    (d / "config.dark.toml").write_text(config_toml("dark"), encoding="utf-8")


if __name__ == "__main__":
    write_configs(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
    print("wrote .streamlit/config.toml (light) and .streamlit/config.dark.toml")
