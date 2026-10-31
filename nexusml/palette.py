"""One palette for the whole application: page chrome, charts, badges.

Chosen for long reading sessions: warm off-white (not pure white) and soft ink (not pure black) in the light theme, a low-glare slate in the dark theme, a single
muted teal accent, and chart colours that stay distinguishable for the common colour-vision deficiencies. Nothing is saturated, nothing glows.
Select the mode with NEXUSML_THEME=light|dark (the matching Streamlit config files are written by `python -m nexusml.ui.theme`).
"""
import os

PALETTES = {
    "light": {"bg": "#F6F4EF", "surface": "#FDFCFA", "sidebar": "#EEEBE3", "border": "#DAD5C8", "text": "#262A31", "muted": "#666B74", "faint": "#8A8F97",
              "accent": "#2F6F6A", "accent_hover": "#255A56", "accent_soft": "#DDEBE8", "clay": "#B2674A", "positive": "#3F7D58", "caution": "#A8781C", "negative": "#A24B4B",
              "positive_soft": "#E1EEE5", "caution_soft": "#F3EAD3", "negative_soft": "#F2E0DE", "grid": "rgba(38,42,49,0.10)"},
    "dark": {"bg": "#14171C", "surface": "#1B1F26", "sidebar": "#101318", "border": "#2B313B", "text": "#E4E2DC", "muted": "#9AA0A8", "faint": "#767C85",
             "accent": "#6CB7AC", "accent_hover": "#85C7BD", "accent_soft": "#1F3331", "clay": "#D08A6C", "positive": "#7DB894", "caution": "#D4A24C", "negative": "#D07C7C",
             "positive_soft": "#1E2E25", "caution_soft": "#332B1A", "negative_soft": "#35201F", "grid": "rgba(228,226,220,0.10)"},
}
CHART = ["#2F6F6A", "#C0714F", "#5B7DB1", "#B8923B", "#7A6C9E", "#6E9C7B", "#A0524F", "#7C8794"]
CHART_DARK = ["#6CB7AC", "#DE9A78", "#86A5D6", "#D9B35F", "#A394C8", "#8FBF9D", "#CC7C79", "#A3AEBA"]


def mode() -> str:
    m = os.environ.get("NEXUSML_THEME", "light").strip().lower()
    return m if m in PALETTES else "light"


def tokens() -> dict:
    return PALETTES[mode()]


def chart_colors() -> list:
    return CHART_DARK if mode() == "dark" else CHART
