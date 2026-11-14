"""
FarmShift - farmer login, saved farms, and NASA POWER data per farm.

Install:  py -m pip install streamlit requests pandas numpy
Run:      py -m streamlit run farmshift_app.py

Data is stored in a local SQLite file (farmshift.db) next to this script.
Passwords are never stored in plain text (PBKDF2-SHA256 with a random salt).
"""

import datetime as dt
import hashlib
import hmac
import os
import re
import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd
import requests
import streamlit as st

DB_PATH = "farmshift.db"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FILL_VALUE = -999
PBKDF2_ROUNDS = 200_000

COMMON_PARAMS = {
    "T2M": "Temperature at 2 m (C)",
    "T2M_MAX": "Max temperature (C)",
    "T2M_MIN": "Min temperature (C)",
    "PRECTOTCORR": "Rainfall (mm/day)",
    "RH2M": "Relative humidity (%)",
    "GWETTOP": "Surface soil wetness (0-1)",
    "GWETROOT": "Root-zone soil wetness (0-1)",
    "ALLSKY_SFC_SW_DWN": "Solar radiation (kWh/m2/day)",
}

st.set_page_config(page_title="FarmShift", page_icon="🌾", layout="wide")


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt BLOB NOT NULL,
                pw_hash BLOB NOT NULL,
                created TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS farms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                crop TEXT,
                area_ha REAL,
                notes TEXT,
                created TEXT NOT NULL,
                UNIQUE (user_id, name)
            );
            """
        )


init_db()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def create_user(username: str, password: str) -> tuple[bool, str]:
    if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
        return False, "Username: 3-30 letters, numbers or underscores."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt = os.urandom(16)
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO users (username, salt, pw_hash, created) VALUES (?, ?, ?, ?)",
                (username, salt, hash_password(password, salt), dt.datetime.now().isoformat()),
            )
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created. You can log in now."


def verify_user(username: str, password: str):
    """Return the user row if the credentials are right, else None."""
    with db() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        hash_password(password, b"0" * 16)  # keep timing similar for unknown users
        return None
    if hmac.compare_digest(row["pw_hash"], hash_password(password, row["salt"])):
        return row
    return None


# --------------------------------------------------------------------------
# Farms
# --------------------------------------------------------------------------
def add_farm(user_id, name, lat, lon, crop, area, notes) -> tuple[bool, str]:
    try:
        with db() as conn:
            conn.execute(
                "INSERT INTO farms (user_id, name, latitude, longitude, crop, area_ha, notes, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, name, lat, lon, crop or None, area or None, notes or None,
                 dt.datetime.now().isoformat()),
            )
    except sqlite3.IntegrityError:
        return False, "You already have a farm with that name."
    return True, "Farm saved."


def list_farms(user_id) -> pd.DataFrame:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, latitude, longitude, crop, area_ha, notes FROM farms "
            "WHERE user_id = ? ORDER BY name",
            (user_id,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows],
                        columns=["id", "name", "latitude", "longitude", "crop", "area_ha", "notes"])


def delete_farm(user_id, farm_id):
    with db() as conn:  # user_id check stops one user deleting another's farm
        conn.execute("DELETE FROM farms WHERE id = ? AND user_id = ?", (farm_id, user_id))


# --------------------------------------------------------------------------
# External APIs
# --------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode(query: str) -> list[dict]:
    resp = requests.get(
        GEO_URL, params={"name": query, "count": 6, "language": "en", "format": "json"}, timeout=30
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


@st.cache_data(ttl=600, show_spinner=False)
def fetch_power(lat: float, lon: float, parameters: tuple, start: str, end: str) -> pd.DataFrame:
    params = {
        "parameters": ",".join(parameters),
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start,
        "end": end,
        "format": "JSON",
    }
    try:
        resp = requests.get(POWER_URL, params=params, timeout=90)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc
    if not resp.ok:
        raise RuntimeError(f"NASA POWER error {resp.status_code}: {resp.text[:500]}")
    block = resp.json()["properties"]["parameter"]
    df = pd.DataFrame(block).replace(FILL_VALUE, np.nan)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    return df


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------
def login_screen():
    st.title("🌾 FarmShift")
    st.write("Log in to see your saved farms and their NASA climate data.")
    tab_login, tab_register = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            user = verify_user(username.strip(), password)
            if user:
                st.session_state["user"] = {"id": user["id"], "username": user["username"]}
                st.rerun()
            else:
                st.error("Wrong username or password.")

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password (8+ characters)", type="password")
            confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account")
        if created:
            if new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                ok, msg = create_user(new_user.strip(), new_pass)
                (st.success if ok else st.error)(msg)


def use_place_callback(place: dict):
    """Runs before the next rerun, so it can safely set the input widgets' values."""
    st.session_state["farm_lat"] = float(place["latitude"])
    st.session_state["farm_lon"] = float(place["longitude"])
    if not st.session_state.get("farm_name"):
        st.session_state["farm_name"] = place["name"]


def add_farm_tab(user_id):
    st.subheader("Add a farm")
    st.session_state.setdefault("farm_lat", 0.0)
    st.session_state.setdefault("farm_lon", 0.0)

    st.markdown("**1. Set the location** - search by place name, or type the coordinates.")
    q_col, b_col = st.columns([4, 1])
    query = q_col.text_input("Search a place", placeholder="e.g. Adoor, Kerala", label_visibility="collapsed")
    if b_col.button("Search", use_container_width=True) and query.strip():
        try:
            st.session_state["geo_results"] = geocode(query.strip())
            if not st.session_state["geo_results"]:
                st.warning("No places found. Try another spelling or type the coordinates.")
        except Exception as exc:
            st.error(f"Place search failed: {exc}")

    results = st.session_state.get("geo_results", [])
    if results:
        labels = [
            f"{r['name']}, {r.get('admin1', '')}, {r.get('country', '')} "
            f"({r['latitude']:.3f}, {r['longitude']:.3f})"
            for r in results
        ]
        idx = st.selectbox("Matches", range(len(results)), format_func=lambda i: labels[i])
        st.button("Use this location", on_click=use_place_callback, args=(results[idx],))

    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", -90.0, 90.0, step=0.0001, format="%.4f", key="farm_lat")
    lon = c2.number_input("Longitude", -180.0, 180.0, step=0.0001, format="%.4f", key="farm_lon")

    st.markdown("**2. Farm details**")
    name = st.text_input("Farm name", key="farm_name")
    d1, d2 = st.columns(2)
    crop = d1.text_input("Main crop (optional)", key="farm_crop")
    area = d2.number_input("Area in hectares (optional)", min_value=0.0, step=0.1, key="farm_area")
    notes = st.text_area("Notes (optional)", key="farm_notes")

    if st.button("Save farm", type="primary"):
        if not name.strip():
            st.error("Give the farm a name.")
        elif lat == 0.0 and lon == 0.0:
            st.error("Set the location first.")
        else:
            ok, msg = add_farm(user_id, name.strip(), lat, lon, crop.strip(), area, notes.strip())
            if ok:
                for k in ("farm_name", "farm_crop", "farm_notes"):
                    st.session_state.pop(k, None)
                st.session_state["flash"] = msg
                st.rerun()  # reload so the new farm shows up in the other tabs
            else:
                st.error(msg)


def my_farms_tab(user_id, farms: pd.DataFrame):
    st.subheader("My farms")
    if farms.empty:
        st.info("No farms yet. Add your first farm in the **Add farm** tab.")
        return
    st.map(farms.rename(columns={"latitude": "lat", "longitude": "lon"}))
    st.dataframe(farms.drop(columns=["id"]), use_container_width=True, hide_index=True)

    with st.expander("Delete a farm"):
        pick = st.selectbox("Farm to delete", farms["id"], format_func=lambda i: farms.set_index("id").loc[i, "name"])
        sure = st.checkbox("Yes, delete this farm permanently")
        if st.button("Delete farm", disabled=not sure):
            delete_farm(user_id, int(pick))
            st.rerun()


def farm_data_tab(farms: pd.DataFrame):
    st.subheader("Farm climate data (NASA POWER)")
    if farms.empty:
        st.info("Add a farm first.")
        return

    farm_id = st.selectbox("Farm", farms["id"], format_func=lambda i: farms.set_index("id").loc[i, "name"])
    farm = farms.set_index("id").loc[farm_id]
    st.caption(f"Location: {farm['latitude']:.4f}, {farm['longitude']:.4f}"
               + (f" | Crop: {farm['crop']}" if farm["crop"] else ""))

    picked = st.multiselect(
        "Parameters", list(COMMON_PARAMS), default=["T2M", "PRECTOTCORR", "GWETROOT"],
        format_func=lambda c: f"{c} - {COMMON_PARAMS[c]}",
    )
    typed = st.text_input("Other parameter codes (comma-separated, optional)")
    parameters = tuple(dict.fromkeys(picked + [p.strip().upper() for p in typed.split(",") if p.strip()]))
    days = st.slider("Days of history", 7, 365, 30)

    if st.button("Get data", type="primary", disabled=not parameters or len(parameters) > 20):
        end = dt.date.today() - dt.timedelta(days=7)  # newest days are often missing
        start = end - dt.timedelta(days=days - 1)
        with st.spinner("Calling NASA POWER..."):
            try:
                df = fetch_power(float(farm["latitude"]), float(farm["longitude"]), parameters,
                                 start.strftime("%Y%m%d"), end.strftime("%Y%m%d"))
                st.session_state["farm_data"] = (farm_id, df)
            except Exception as exc:
                st.session_state.pop("farm_data", None)
                st.error(str(exc))

    saved = st.session_state.get("farm_data")
    if saved and saved[0] == farm_id:
        df = saved[1]
        cols = st.columns(min(len(df.columns), 4))
        for i, name in enumerate(df.columns):
            s = df[name].dropna()
            if s.empty:
                continue
            total = name == "PRECTOTCORR"
            with cols[i % len(cols)]:
                st.metric(f"{name} ({'total' if total else 'average'})",
                          f"{(s.sum() if total else s.mean()):.2f}")
        for name in df.columns:
            st.markdown(f"**{name}** - {COMMON_PARAMS.get(name, '')}")
            st.line_chart(df[name])
        st.download_button("Download CSV", df.to_csv().encode(), "farm_data.csv", "text/csv")
        st.caption("Missing values (-999) are shown as gaps. Source: NASA POWER.")


def main_screen():
    user = st.session_state["user"]
    with st.sidebar:
        st.write(f"Logged in as **{user['username']}**")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()

    st.title("🌾 FarmShift")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    farms = list_farms(user["id"])
    tab_farms, tab_add, tab_data = st.tabs(["My farms", "Add farm", "Farm data"])
    with tab_farms:
        my_farms_tab(user["id"], farms)
    with tab_add:
        add_farm_tab(user["id"])
    with tab_data:
        farm_data_tab(farms)


if "user" in st.session_state:
    main_screen()
else:
    login_screen()
