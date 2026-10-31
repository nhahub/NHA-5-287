"""Application shell: page setup, sign-in, navigation and the error boundary around every page."""
import contextlib
import hmac
import os
import re
import time
from pathlib import Path
from typing import Optional, Tuple

from .. import config, jobs, runstore
from . import legacy, nav, state, theme
from ._st import ss, st
from .pages import analyze, assistant, data, home, models, runs, settings


def _assets() -> Tuple[Optional[str], Optional[str]]:
    """A favicon and a wordmark generated from the app name and palette (so renaming the app renames them)."""
    d = Path(config.artifacts_dir()) / "ui"
    try:
        d.mkdir(parents=True, exist_ok=True)
        from PIL import Image, ImageDraw, ImageFont
        from .. import palette
        t = palette.tokens()
        icon = d / f"favicon-{theme.APP_NAME[:1].upper()}-{palette.mode()}.png"
        if not icon.exists():
            im = Image.new("RGBA", (96, 96), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            dr.rounded_rectangle((4, 4, 92, 92), radius=22, fill=t["accent"])
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
            except OSError:
                font = ImageFont.load_default()
            dr.text((48, 50), theme.APP_NAME[:1].upper(), fill="#FFFFFF" if palette.mode() == "light" else "#0E1514", font=font, anchor="mm")
            im.save(icon)
        logo = d / f"logo-{re.sub(r'[^A-Za-z0-9]', '', theme.APP_NAME)}-{palette.mode()}.svg"
        if not logo.exists():
            logo.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="190" height="40" viewBox="0 0 190 40"><rect x="2" y="6" width="28" height="28" rx="8" fill="{t["accent"]}"/>'
                            f'<text x="16" y="26" text-anchor="middle" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="17" font-weight="700" fill="{"#fff" if palette.mode() == "light" else "#0E1514"}">{theme.APP_NAME[:1].upper()}</text>'
                            f'<text x="40" y="27" font-family="Inter,Segoe UI,Arial,sans-serif" font-size="18" font-weight="650" fill="{t["text"]}">{theme.APP_NAME}</text></svg>', encoding="utf-8")
        return str(icon), str(logo)
    except Exception:
        return None, None


def _gate() -> bool:
    """Optional shared-password sign-in (set NEXUSML_APP_PASSWORD). The name entered becomes the person's own workspace."""
    pw = os.environ.get("NEXUSML_APP_PASSWORD")
    if not pw or ss.get("authed"):
        return True
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.write("")
        theme.brand_block()
        with st.container(border=True):
            st.markdown("### Sign in")
            name = st.text_input("Your name", key="gate-name", placeholder="How should we address you?")
            pwd = st.text_input("Password", type="password", key="gate-pw")
            if st.button("Continue", type="primary", use_container_width=True, key="gate-go"):
                if hmac.compare_digest(pwd.encode(), pw.encode()):
                    ss.authed = True
                    ss.user = re.sub(r"[^\w .\-]", "", name).strip()[:40] or "user"
                    runstore.set_user(ss.user)
                    st.rerun()
                ss.fails = int(ss.get("fails", 0)) + 1
                time.sleep(min(4, ss.fails))                         # slows guessing without locking anyone out
                st.error("That password is not correct.")
    return False


@contextlib.contextmanager
def error_boundary(where: str):
    """A page that fails shows a short message and keeps the rest of the app usable; the details go to the error log."""
    try:
        yield
    except Exception as e:
        if any(w in type(e).__name__.lower() for w in ("rerun", "stop", "script")):
            raise
        legacy.log_error(where, e)
        st.error("Something went wrong on this page. Your data and conversations are safe.")
        with st.expander("Technical details"):
            st.code(f"{type(e).__name__}: {str(e)[:400]}")
            st.caption(f"Saved to {Path(config.artifacts_dir()) / 'app_errors.log'}")


def _pages():
    P = st.Page
    page = {"home": P(home.render, title="Home", icon=":material/home:", url_path="home", default=True),
            "assistant": P(assistant.render, title="Assistant", icon=":material/forum:", url_path="assistant"),
            "data": P(data.render, title="Data", icon=":material/table_chart:", url_path="data"),
            "analyze": P(analyze.render, title="New analysis", icon=":material/add_circle:", url_path="new-analysis"),
            "history": P(runs.render, title="Analyses", icon=":material/history:", url_path="analyses"),
            "models": P(models.render, title="Models", icon=":material/deployed_code:", url_path="models"),
            "settings": P(settings.render, title="Settings", icon=":material/settings:", url_path="settings")}
    for k, v in page.items():
        nav.register(k, v)
    return {"Workspace": [page["home"], page["assistant"], page["data"]], "Analyses": [page["analyze"], page["history"]], "Models": [page["models"]], "System": [page["settings"]]}


def main() -> None:
    icon, logo = _assets()
    st.set_page_config(page_title=theme.APP_NAME, page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    theme.apply()
    if not _gate():
        return
    state.init()
    if logo:
        st.logo(logo, size="large")
    navigation = st.navigation(_pages(), position="sidebar")
    with st.sidebar:
        live = [j for j in jobs.active() if j["status"] in ("running", "queued")]
        if live:
            st.caption(f"{len(live)} analysis running")
        if os.environ.get("NEXUSML_APP_PASSWORD"):
            st.caption(f"Signed in as {ss.user}")
            if st.button("Sign out", key="side-out"):
                for k in ("authed", "user", "messages", "conv_id", "ds_id", "job_id"):
                    ss.pop(k, None)
                st.rerun()
    with error_boundary("page"):
        navigation.run()
