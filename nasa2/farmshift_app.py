"""
FarmShift - save your farms on a satellite map and see NASA climate data for each one.
No login needed. Farms are kept for the current browser session; download/upload them as CSV.

Install:  py -m pip install streamlit requests pandas numpy folium streamlit-folium
Run:      py -m streamlit run farmshift_app.py
"""

import datetime as dt
import html
import math

import folium
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FILL_VALUE = -999
FARM_COLUMNS = ["id", "name", "latitude", "longitude", "crop", "area_ha", "notes"]

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
# Farms (kept in the visitor's session - no accounts, no database)
# --------------------------------------------------------------------------
def _store() -> list:
    if "farms" not in st.session_state:
        st.session_state["farms"] = [
            dict(id=1, name="Sample farm (Kottayam)", latitude=9.59, longitude=76.52, crop="Rice",
                 area_ha=2.0, notes="A sample farm so you can try the app. Delete it any time.")
        ]
        st.session_state["next_id"] = 2
    return st.session_state["farms"]


def add_farm(name, lat, lon, crop, area, notes) -> tuple:
    store = _store()
    if any(f["name"].lower() == name.lower() for f in store):
        return False, "You already have a farm with that name."
    store.append(dict(id=st.session_state["next_id"], name=name, latitude=float(lat), longitude=float(lon),
                      crop=crop or None, area_ha=area or None, notes=notes or None))
    st.session_state["next_id"] += 1
    return True, "Farm saved."


def list_farms() -> pd.DataFrame:
    df = pd.DataFrame(_store(), columns=FARM_COLUMNS)
    return df.sort_values("name").reset_index(drop=True)


def delete_farm(farm_id):
    st.session_state["farms"] = [f for f in _store() if f["id"] != farm_id]


def import_farms_csv(uploaded) -> int:
    """Add farms from a CSV (columns: name, latitude, longitude, crop, area_ha, notes). Returns how many were added."""
    df = pd.read_csv(uploaded)
    added = 0
    for _, r in df.iterrows():
        try:
            if pd.isna(r["name"]):
                continue
            name, lat, lon = str(r["name"]).strip(), float(r["latitude"]), float(r["longitude"])
        except (KeyError, ValueError):
            continue
        if not name or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        crop = r.get("crop")
        notes = r.get("notes")
        area = r.get("area_ha")
        ok, _msg = add_farm(
            name, lat, lon,
            crop if isinstance(crop, str) else "",
            float(area) if pd.notna(area) else None,
            notes if isinstance(notes, str) else "",
        )
        added += int(ok)
    return added


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
# Satellite maps
# --------------------------------------------------------------------------
ESRI_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_LABELS = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/"
    "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
)
GIBS_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "MODIS_Terra_CorrectedReflectance_TrueColor/default/{date}/"
    "GoogleMapsCompatible_Level9/{{z}}/{{y}}/{{x}}.jpg"
)


def base_map(center, zoom) -> folium.Map:
    """Satellite map with switchable layers (Esri imagery, NASA MODIS, street map)."""
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True)
    folium.TileLayer(
        ESRI_IMAGERY, name="Satellite (Esri)", max_zoom=19, show=True,
        attr="Imagery: Esri, Maxar, Earthstar Geographics",
    ).add_to(m)
    folium.TileLayer(
        ESRI_LABELS, name="Place names", overlay=True, control=True, show=True, attr="Labels: Esri",
    ).add_to(m)
    gibs_date = (dt.date.today() - dt.timedelta(days=3)).isoformat()
    folium.TileLayer(
        GIBS_URL.format(date=gibs_date), name=f"NASA MODIS true colour ({gibs_date})",
        max_native_zoom=9, max_zoom=12, show=False, attr="Imagery: NASA GIBS / EOSDIS",
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map", show=False).add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


def add_farm_markers(m: folium.Map, farms: pd.DataFrame):
    """Drop a pin on the map for every saved farm."""
    for _, f in farms.iterrows():
        lines = [f"<b>{html.escape(f['name'])}</b>"]
        if isinstance(f["crop"], str) and f["crop"]:
            lines.append(f"Crop: {html.escape(f['crop'])}")
        if pd.notna(f["area_ha"]):
            lines.append(f"Area: {f['area_ha']:g} ha")
        lines.append(f"{f['latitude']:.4f}, {f['longitude']:.4f}")
        folium.Marker(
            [f["latitude"], f["longitude"]],
            tooltip=html.escape(f["name"]),
            popup=folium.Popup("<br>".join(lines), max_width=250),
            icon=folium.Icon(color="green", icon="leaf", prefix="fa"),
        ).add_to(m)


def fit_view(farms: pd.DataFrame):
    """Centre and zoom that roughly fit all the farms on screen."""
    lat, lon = farms["latitude"], farms["longitude"]
    span = max(lat.max() - lat.min(), lon.max() - lon.min(), 0.005) * 1.6
    zoom = max(3, min(15, int(math.log2(360 / span))))
    return [float(lat.mean()), float(lon.mean())], zoom


def nearest_farm(farms: pd.DataFrame, clicked: dict | None):
    """Read a clicked pin: return the saved farm at those coordinates."""
    if not clicked or farms.empty:
        return None
    d = (farms["latitude"] - clicked["lat"]) ** 2 + (farms["longitude"] - clicked["lng"]) ** 2
    i = d.idxmin()
    return farms.loc[i] if d.loc[i] < 0.001 ** 2 else None


# --------------------------------------------------------------------------
# Screens
# --------------------------------------------------------------------------
def use_place_callback(place: dict):
    """Runs before the next rerun, so it can safely set the input widgets' values."""
    st.session_state["farm_lat"] = float(place["latitude"])
    st.session_state["farm_lon"] = float(place["longitude"])
    st.session_state["add_view"] = ([float(place["latitude"]), float(place["longitude"])], 13)
    if not st.session_state.get("farm_name"):
        st.session_state["farm_name"] = place["name"]


def add_farm_tab(farms: pd.DataFrame):
    st.subheader("Add a farm")
    st.session_state.setdefault("farm_lat", 0.0)
    st.session_state.setdefault("farm_lon", 0.0)

    st.markdown("**1. Set the location** - click the satellite map to drop a pin, search a place, or type coordinates.")
    q_col, b_col = st.columns([4, 1])
    query = q_col.text_input("Search a place", placeholder="e.g. Adoor, Kerala", label_visibility="collapsed")
    if b_col.button("Search", use_container_width=True) and query.strip():
        try:
            st.session_state["geo_results"] = geocode(query.strip())
            if not st.session_state["geo_results"]:
                st.warning("No places found. Try another spelling, or click the map.")
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

    # --- interactive satellite map (drawn BEFORE the lat/lon inputs so a click can set them)
    lat0, lon0 = st.session_state["farm_lat"], st.session_state["farm_lon"]
    has_pin = not (lat0 == 0.0 and lon0 == 0.0)
    view = st.session_state.get("add_view")
    if view is None:
        if has_pin:
            view = ([lat0, lon0], 13)
        elif not farms.empty:
            view = fit_view(farms)
        else:
            view = ([20.0, 78.0], 4)

    m = base_map(*view)
    add_farm_markers(m, farms)  # your existing farms, for context
    if has_pin:
        folium.Marker(
            [lat0, lon0], tooltip="New farm location", icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
        ).add_to(m)
    out = st_folium(
        m, height=460, use_container_width=True, key="map_add",
        returned_objects=["last_clicked", "center", "zoom"],
    )

    click = (out or {}).get("last_clicked")
    if click and st.session_state.get("last_click") != (click["lat"], click["lng"]):
        st.session_state["last_click"] = (click["lat"], click["lng"])
        st.session_state["farm_lat"] = round(click["lat"], 6)
        st.session_state["farm_lon"] = round(click["lng"], 6)
        st.session_state["add_view"] = ([click["lat"], click["lng"]], (out or {}).get("zoom") or 13)
        st.rerun()  # redraw with the new red pin
    elif out and out.get("center") and out.get("zoom"):
        st.session_state["add_view"] = ([out["center"]["lat"], out["center"]["lng"]], out["zoom"])

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
            ok, msg = add_farm(name.strip(), lat, lon, crop.strip(), area, notes.strip())
            if ok:
                for k in ("farm_name", "farm_crop", "farm_notes", "last_click"):
                    st.session_state.pop(k, None)
                st.session_state["flash"] = msg
                st.rerun()  # reload so the new farm shows up in the other tabs
            else:
                st.error(msg)


def climate_panel(farm_id, farm, prefix: str):
    """NASA POWER controls, charts and numbers for one farm. Used by both the pin view and the data tab."""
    k = f"{prefix}_{farm_id}"
    st.caption(
        f"Location: {farm['latitude']:.4f}, {farm['longitude']:.4f}"
        + (f" | Crop: {farm['crop']}" if isinstance(farm["crop"], str) and farm["crop"] else "")
    )
    picked = st.multiselect(
        "Parameters", list(COMMON_PARAMS), default=["T2M", "PRECTOTCORR", "GWETROOT"],
        format_func=lambda c: f"{c} - {COMMON_PARAMS[c]}", key=f"{k}_pick",
    )
    typed = st.text_input("Other parameter codes (comma-separated, optional)", key=f"{k}_typed")
    parameters = tuple(dict.fromkeys(picked + [p.strip().upper() for p in typed.split(",") if p.strip()]))
    days = st.slider("Days of history", 7, 365, 30, key=f"{k}_days")

    if st.button("Get NASA data", type="primary", key=f"{k}_go", disabled=not parameters or len(parameters) > 20):
        end = dt.date.today() - dt.timedelta(days=7)  # newest days are often missing
        start = end - dt.timedelta(days=days - 1)
        with st.spinner("Calling NASA POWER..."):
            try:
                st.session_state[f"{k}_data"] = fetch_power(
                    float(farm["latitude"]), float(farm["longitude"]), parameters,
                    start.strftime("%Y%m%d"), end.strftime("%Y%m%d"),
                )
            except Exception as exc:
                st.session_state.pop(f"{k}_data", None)
                st.error(str(exc))

    df = st.session_state.get(f"{k}_data")
    if df is not None:
        cols = st.columns(min(len(df.columns), 4))
        for i, name in enumerate(df.columns):
            s = df[name].dropna()
            if s.empty:
                continue
            total = name == "PRECTOTCORR"
            with cols[i % len(cols)]:
                st.metric(f"{name} ({'total' if total else 'average'})", f"{(s.sum() if total else s.mean()):.2f}")
        for name in df.columns:
            st.markdown(f"**{name}** - {COMMON_PARAMS.get(name, '')}")
            st.line_chart(df[name])
        st.download_button("Download CSV", df.to_csv().encode(), "farm_data.csv", "text/csv", key=f"{k}_csv")
        st.caption("Missing values (-999) are shown as gaps. Source: NASA POWER.")


def my_farms_tab(farms: pd.DataFrame):
    st.subheader("My farms")
    if farms.empty:
        st.info("No farms yet. Add your first farm in the **Add farm** tab.")
        return

    center, zoom = fit_view(farms)
    m = base_map(center, zoom)
    add_farm_markers(m, farms)
    out = st_folium(
        m, height=500, use_container_width=True, key="map_farms",
        returned_objects=["last_object_clicked"],
    )
    st.caption("Click a green pin to open that farm. Use the layer button (top right of the map) to switch imagery.")

    farm = nearest_farm(farms, (out or {}).get("last_object_clicked"))
    if farm is not None:
        st.markdown(f"### 📍 {farm['name']}")
        if isinstance(farm["notes"], str) and farm["notes"]:
            st.write(farm["notes"])
        if pd.notna(farm["area_ha"]):
            st.write(f"Area: {farm['area_ha']:g} ha")
        climate_panel(int(farm["id"]), farm, "pin")

    st.dataframe(farms.drop(columns=["id"]), use_container_width=True, hide_index=True)

    with st.expander("Save or load your farms (CSV)"):
        st.write("Farms are kept while this page stays open. Download them to keep a copy, and upload the file later to restore them.")
        st.download_button("Download my farms (CSV)", farms.drop(columns=["id"]).to_csv(index=False).encode(),
                           "my_farms.csv", "text/csv")
        up = st.file_uploader("Upload a farms CSV", type="csv")
        if up is not None:
            try:
                added = import_farms_csv(up)
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")
            else:
                if added:
                    st.session_state["flash"] = f"Added {added} farm(s) from the file."
                    st.rerun()

    with st.expander("Delete a farm"):
        pick = st.selectbox("Farm to delete", farms["id"], format_func=lambda i: farms.set_index("id").loc[i, "name"])
        sure = st.checkbox("Yes, delete this farm permanently")
        if st.button("Delete farm", disabled=not sure):
            delete_farm(int(pick))
            st.rerun()


def farm_data_tab(farms: pd.DataFrame):
    st.subheader("Farm climate data (NASA POWER)")
    if farms.empty:
        st.info("Add a farm first.")
        return
    farm_id = st.selectbox("Farm", farms["id"], format_func=lambda i: farms.set_index("id").loc[i, "name"])
    farm = farms.set_index("id").loc[farm_id]
    climate_panel(int(farm_id), farm, "tab")


def main():
    with st.sidebar:
        st.markdown("### 🌾 FarmShift")
        st.caption("No login needed. Your farms stay in this browser session. "
                   "Use **My farms > Save or load your farms** to keep a copy.")

    st.title("🌾 FarmShift")
    st.write("Pin your farms on a satellite map and see NASA climate data for each one.")
    if "flash" in st.session_state:
        st.success(st.session_state.pop("flash"))
    farms = list_farms()
    tab_farms, tab_add, tab_data = st.tabs(["My farms", "Add farm", "Farm data"])
    with tab_farms:
        my_farms_tab(farms)
    with tab_add:
        add_farm_tab(farms)
    with tab_data:
        farm_data_tab(farms)


main()