from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from dotenv import load_dotenv

from models.hybrid.hybrid_planner import HybridSolver
from models.llm.llm_constraint_extractor import LLMConstraintExtractor

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(
    page_title="SeaRouteGPT – AI-Powered Cruise Planning",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "SeaRouteGPT: Hybrid LLM+MILP Cruise Planning Engine"},
)

LOCAL_CATALOG_PATH = Path("data/raw/cruises.json")

DEFAULT_API_URL = os.getenv("RAPIDAPI_URL", "https://cruise-api1.p.rapidapi.com/cruises/search")
DEFAULT_HOST = os.getenv("RAPIDAPI_HOST", "cruise-api1.p.rapidapi.com")


# ----------------------------
# STYLING
# ----------------------------
def inject_premium_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Syne:wght@400;500;600;700&display=swap');
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }

        .stApp{
            background:
              radial-gradient(1200px 700px at 20% 10%, rgba(56,189,248,0.10), transparent 55%),
              radial-gradient(1000px 700px at 80% 20%, rgba(3,102,214,0.10), transparent 55%),
              linear-gradient(135deg, #070b14 0%, #071424 45%, #070b14 100%);
            background-attachment: fixed;
        }

        h1,h2,h3{ font-family:'Syne',sans-serif !important; letter-spacing:-0.02em; }
        #MainMenu{visibility:hidden;} footer{visibility:hidden;} header{visibility:hidden;}

        section[data-testid="stSidebar"]{
            background: linear-gradient(180deg, rgba(7, 11, 20, 0.92), rgba(7, 20, 36, 0.92)) !important;
            border-right: 1px solid rgba(125, 211, 252, 0.12) !important;
            backdrop-filter: blur(12px);
        }

        .hero{
            margin: 0 -1rem 1.75rem -1rem;
            padding: 2.4rem 2rem 2rem 2rem;
            border: 1px solid rgba(125, 211, 252, 0.12);
            background: linear-gradient(135deg, rgba(3,102,214,0.14), rgba(56,189,248,0.08));
            border-radius: 1.35rem;
            position: relative;
            overflow: hidden;
        }
        .hero::after{
            content:'';
            position:absolute;
            inset:-220px -220px auto auto;
            width:520px; height:520px;
            background: radial-gradient(circle, rgba(56,189,248,0.12), transparent 70%);
            transform: rotate(18deg);
        }
        .hero-title{
            position:relative; z-index:1;
            font-size:3rem; font-weight:900;
            color:#fff; margin:0; line-height:1.05;
            font-family:'Syne',sans-serif;
        }
        .hero-sub{
            position:relative; z-index:1;
            color: rgba(255,255,255,0.72);
            font-size:1.05rem; margin-top:0.75rem; max-width:70ch;
        }
        .badge{
            position:relative; z-index:1;
            display:inline-flex; gap:0.5rem; align-items:center;
            padding:0.45rem 0.9rem;
            border-radius:999px;
            border:1px solid rgba(125,211,252,0.25);
            background: rgba(125,211,252,0.10);
            color:#7dd3fc;
            font-weight:800; font-size:0.85rem;
            margin-top:1.05rem;
        }

        .stTextArea textarea, .stTextInput input, .stNumberInput input, .stSelectbox select{
            background: rgba(15,23,42,0.45) !important;
            border: 1px solid rgba(125,211,252,0.18) !important;
            border-radius: 0.95rem !important;
            color: #fff !important;
            padding: 0.95rem 1rem !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus{
            border-color: rgba(125,211,252,0.45) !important;
            box-shadow: 0 0 0 3px rgba(125,211,252,0.10) !important;
        }

        div.stButton > button{
            width:100%;
            padding:0.95rem 1.15rem !important;
            border-radius:0.95rem !important;
            border:1px solid rgba(125,211,252,0.28) !important;
            background: linear-gradient(135deg,#0366d6 0%, #0284c7 100%) !important;
            color:#fff !important;
            font-weight:900 !important;
            letter-spacing:0.35px;
            text-transform:uppercase;
            box-shadow: 0 14px 34px rgba(3,102,214,0.25);
            transition: all 0.2s ease !important;
            font-family:'Syne',sans-serif !important;
        }
        div.stButton > button:hover{ transform: translateY(-2px) !important; }

        .result{
            border-radius:1.4rem;
            border: 1px solid rgba(125,211,252,0.18);
            background: linear-gradient(135deg, rgba(3,102,214,0.16), rgba(56,189,248,0.08));
            padding:1.7rem;
            box-shadow: 0 18px 56px rgba(0,0,0,0.22);
        }
        .kicker{
            color: rgba(125,211,252,0.90);
            font-weight:900;
            letter-spacing:0.08em;
            text-transform:uppercase;
            font-size:0.8rem;
            margin-bottom:0.35rem;
        }
        .big{
            font-family:'Syne',sans-serif;
            font-size:2rem;
            font-weight:900;
            color:#fff;
            line-height:1.2;
            margin:0 0 0.75rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">🧭 SeaRouteGPT</div>
          <div class="hero-sub">Elegant cruise recommendations powered by constraint extraction + optimization.</div>
          <div class="badge">⚡ Hybrid LLM + MILP Planner</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# LOCAL CATALOG
# ----------------------------
def load_local_catalog(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    if isinstance(data, list):
        return data
    return []


# ----------------------------
# RAPIDAPI (POST /cruises/search)
# ----------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_cruises_search_cached(
    api_url: str,
    host: str,
    api_key: str,
    payload: Dict[str, Any],
    *,
    max_pages: int = 10,
    page_size: int = 10,
    timeout_s: int = 15,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host,
        "Content-Type": "application/json",
    }

    all_cruises: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    last_error: Optional[str] = None

    page = 1
    while page <= max_pages:
        payload_page = dict(payload)
        payload_page["page"] = page
        payload_page["pageSize"] = page_size

        try:
            r = requests.post(api_url, headers=headers, json=payload_page, timeout=timeout_s)

            if r.status_code == 429:
                wait = min(12, 2 ** (page - 1))
                time.sleep(wait)
                continue

            if r.status_code != 200:
                last_error = f"HTTP {r.status_code}: {r.text[:400]}"
                break

            data = r.json()
            cruises = data.get("data", []) if isinstance(data, dict) else []
            total_pages = int(data.get("total_pages", 1)) if isinstance(data, dict) else 1

            for c in cruises:
                cid = c.get("cruiseId")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_cruises.append(c)

            if page >= total_pages:
                break

            page += 1
            time.sleep(0.6)
        except Exception as e:
            last_error = f"Fetch error: {str(e)}"
            break

    return all_cruises, last_error


# ----------------------------
# CONSTRAINTS -> PAYLOAD
# ----------------------------
def build_payload_from_constraints(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build RapidAPI /cruises/search payload from extracted HARD constraints.
    Keep it moderately broad; MILP does the final filtering.
    """
    hard = extracted.get("hard_constraints", {}) or {}

    # Dates (default to 2026 if user didn't specify)
    dw = hard.get("departure_date_window")
    if dw and dw.get("earliest") and dw.get("latest"):
        earliest = str(dw["earliest"])
        latest = str(dw["latest"])
    else:
        earliest = "2026-01-01"
        latest = "2026-12-31"

    # Duration
    dr = hard.get("duration_range")
    duration_min = int(dr["min_days"]) if (dr and dr.get("min_days") is not None) else 3
    duration_max = int(dr["max_days"]) if (dr and dr.get("max_days") is not None) else 30

    # Guests (API expects array!)
    num_guests = hard.get("num_guests") or 2
    number_of_guests = [int(num_guests)]

    # Destinations (API allows max 2)
    dests = hard.get("allowed_destinations")
    if isinstance(dests, list):
        destinations = dests[:2]
    else:
        destinations = []

    payload = {
        "destinations": destinations,
        "earliestStartDate": earliest,
        "latestStartDate": latest,
        "durationMin": duration_min,
        "durationMax": duration_max,
        "numberOfGuests": number_of_guests,
    }

    return payload


def extract_constraints_ui(user_request: str, request_id: str) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"error": "GOOGLE_API_KEY not found in environment."}

    extractor = LLMConstraintExtractor(api_key=api_key)
    return extractor.extract_constraints(user_request=user_request, request_id=request_id)


# ----------------------------
# SOLVER
# ----------------------------
def run_hybrid(
    user_request: str,
    cruises: List[Dict[str, Any]],
    *,
    alpha: float,
    beta: float,
    request_id: str,
    time_limit: int,
) -> Dict[str, Any]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {
            "status": "error",
            "error_type": "missing_google_api_key",
            "message": "GOOGLE_API_KEY not found in environment.",
        }

    try:
        solver = HybridSolver(api_key=api_key)
        return solver.solve(
            user_request=user_request,
            cruises=cruises,
            alpha=alpha,
            beta=beta,
            request_id=request_id,
            time_limit_seconds=int(time_limit),
        )
    except Exception as e:
        return {"status": "error", "error_type": "solver_error", "message": str(e)}


# ----------------------------
# HELPERS (display)
# ----------------------------
def safe_list(x: Any) -> List[str]:
    return [str(i) for i in x] if isinstance(x, list) else []


def format_money(x: Any) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x)


def guess_url(cruise: Dict[str, Any]) -> str:
    return (
        cruise.get("itineraryUrl")
        or cruise.get("bookingUrl")
        or cruise.get("url")
        or cruise.get("link")
        or ""
    )


# ----------------------------
# UI
# ----------------------------
def render_sidebar() -> Dict[str, Any]:
    with st.sidebar:
        st.markdown("### ⚙️ Controls")

        alpha = st.slider("Cost Weight (α)", 0.0, 1.0, 0.6, 0.05)
        beta = 1.0 - alpha
        time_limit = st.slider("MILP Time Limit (sec)", 5, 60, 10, 5)

        st.divider()
        st.markdown("### 📡 Data Source")
        mode = st.radio("Cruise data", ["LIVE (RapidAPI)", "LOCAL (cruises.json)"], index=0)

        st.divider()
        st.markdown("### 🔌 RapidAPI Settings")

        api_url = st.text_input("Search endpoint", value=DEFAULT_API_URL)
        host = st.text_input("X-RapidAPI-Host", value=DEFAULT_HOST)

        max_pages = st.slider("Max pages (10 cruises/page)", 1, 10, 10, 1)

        rapid_key = os.getenv("RAPIDAPI_KEY", "")
        if not rapid_key and mode.startswith("LIVE"):
            st.warning("RAPIDAPI_KEY missing in .env", icon="🔑")

        st.caption("This UI uses POST /cruises/search (same as your cache_cruises.py).")

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "time_limit": int(time_limit),
            "mode": mode,
            "api_url": api_url,
            "host": host,
            "max_pages": int(max_pages),
            "rapid_key": rapid_key,
        }


def render_input() -> str:
    st.markdown("### ✍️ Your request")
    st.caption("Be specific about destination, dates, duration, and budget.")
    return st.text_area(
        "Cruise Request",
        height=190,
        placeholder="Example: 7-day Caribbean cruise in June 2026 for 2 people under $3000. Prefer Bahamas and Cayman Islands.",
        label_visibility="collapsed",
    )


def debug_solver_result(result: Dict[str, Any], cruises: List[Dict[str, Any]]) -> None:
    st.markdown("### 🐛 Debug Information")

    with st.expander("Extracted Constraints (These filtered your cruises)", expanded=True):
        constraints = result.get("constraints_extracted", {})
        st.json(constraints)
        st.markdown(
            """
        **These hard constraints were extracted from your request.**
        If no cruises match, one or more constraints may be:
        - **Too strict** (e.g., exact dates, very low budget)
        - **Mis-extracted** (e.g., wrong destination code)
        - **Data mismatch** (API returns missing prices or unexpected destination codes)
        """
        )

    with st.expander("Full Solver Response", expanded=False):
        st.json(result)


def render_result(
    result: Dict[str, Any],
    fetch_debug: Optional[str] = None,
    cruises: Optional[List[Dict[str, Any]]] = None,
) -> None:
    if fetch_debug:
        with st.expander("Show fetch error details", expanded=False):
            st.code(fetch_debug)

    # Only show this "no feasible" info when NOT success
    if result.get("status") != "success" and result.get("constraints_extracted"):
        st.info("No feasible cruises. Here are the extracted constraints:")
        st.json(result.get("constraints_extracted", {}))
        if cruises:
            debug_solver_result(result, cruises)

    if result.get("status") != "success" or not result.get("selected_cruise"):
        msg = result.get("message") or "No cruise found. Try changing budget/dates/destination."
        st.error(f"🚫 {msg}")
        return

    cruise = result["selected_cruise"]

    name = cruise.get("name") or cruise.get("cruiseName") or "Recommended Cruise"
    dep = cruise.get("departureDate", "N/A")
    dur = cruise.get("duration", "N/A")
    price = cruise.get("roomPriceWithTaxesFees") or cruise.get("price") or 0
    cruise_id = cruise.get("cruiseId", "N/A")

    destinations = safe_list(cruise.get("itineraryDestinations") or cruise.get("destinations") or [])
    ports = safe_list(cruise.get("itineraryPorts") or cruise.get("ports") or [])
    url = guess_url(cruise)

    st.markdown('<div class="kicker">Recommended</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="big">{name}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("📅 Departure", str(dep))
    c2.metric("⏱️ Duration", f"{dur} days" if str(dur).isdigit() else str(dur))
    c3.metric("💰 Price", format_money(price))

    st.markdown(f"**Cruise ID:** `{cruise_id}`")
    if destinations:
        st.markdown("**🗺️ Destinations:** " + ", ".join(destinations))
    if ports:
        st.markdown("**⚓ Ports:** " + ", ".join(ports))

    if url:
        st.link_button("🎫 View / Book this cruise", url)
    else:
        st.info("Booking URL not available for this result.")

    with st.expander("🔍 Constraint extraction & solver details", expanded=False):
        if "constraints_extracted" in result:
            st.markdown("#### 🎯 Hard Constraints")
            st.json(result.get("constraints_extracted", {}))
        if "preferences_extracted" in result:
            st.markdown("#### ♥️ Soft Preferences")
            st.json(result.get("preferences_extracted", {}))
        st.markdown("#### Raw Result (minus selected_cruise)")
        st.json({k: v for k, v in result.items() if k != "selected_cruise"})


def main() -> None:
    inject_premium_css()
    render_hero()

    sidebar = render_sidebar()
    user_request = render_input()

    run_col = st.columns([1, 2, 1])[1]
    with run_col:
        run = st.button("⚓ Plan My Cruise")

    if not run:
        return

    if not user_request.strip():
        st.warning("Please enter a request first.")
        return

    cruises: List[Dict[str, Any]] = []
    fetch_err: Optional[str] = None

    # Stable request id for BOTH: extraction + solver (so caches line up)
    ui_request_id = f"ui_req_{int(time.time())}"

    if sidebar["mode"].startswith("LIVE"):
        if not sidebar["rapid_key"]:
            st.error("RAPIDAPI_KEY missing in your .env. Add it and restart Streamlit.")
            return

        with st.spinner("🧠 Extracting constraints (LLM)..."):
            extracted = extract_constraints_ui(user_request.strip(), ui_request_id)

        if "error" in extracted:
            st.error(extracted["error"])
            return

        payload = build_payload_from_constraints(extracted)

        with st.expander("🔍 Debug: extracted constraints & API payload", expanded=False):
            st.json(extracted)
            st.json(payload)

        with st.spinner("📡 Fetching live cruise candidates (POST /cruises/search)..."):
            cruises, fetch_err = fetch_cruises_search_cached(
                api_url=sidebar["api_url"],
                host=sidebar["host"],
                api_key=sidebar["rapid_key"],
                payload=payload,
                max_pages=sidebar["max_pages"],
            )

        if not cruises:
            local = load_local_catalog(LOCAL_CATALOG_PATH)
            if local:
                st.warning("Live API fetch failed — falling back to local catalog.")
                cruises = local
            else:
                st.error("No cruise data available right now. Please try again later.")
                render_result({"status": "error", "message": "No data"}, fetch_debug=fetch_err, cruises=[])
                return
    else:
        cruises = load_local_catalog(LOCAL_CATALOG_PATH)
        if not cruises:
            st.error("Local catalog not found or empty: data/raw/cruises.json")
            return

    st.success(f"🚢 Loaded {len(cruises)} cruises")

    with st.spinner("🔄 Optimizing with Hybrid (LLM + MILP)..."):
        result = run_hybrid(
            user_request=user_request.strip(),
            cruises=cruises,
            alpha=sidebar["alpha"],
            beta=sidebar["beta"],
            request_id=ui_request_id,
            time_limit=sidebar["time_limit"],
        )

    render_result(result, fetch_debug=fetch_err, cruises=cruises)


if __name__ == "__main__":
    main()