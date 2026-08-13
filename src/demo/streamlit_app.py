"""
Streamlit demo for the AI-Powered Trade Route Recommendation System.

Calls the running FastAPI backend (src/api/main.py) for port search and route
recommendations, and visualizes results as a table + interactive map.

Run with (make sure the FastAPI backend is already running in another terminal):
    streamlit run src/demo/streamlit_app.py
"""

import requests
import streamlit as st
import plotly.graph_objects as go

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Trade Route Recommender", layout="wide")

st.title("🚢 AI-Powered Trade Route Recommendation System")
st.caption(
    "Recommends ranked alternative maritime routes under disruption scenarios "
    "(e.g. Red Sea / Suez Canal crisis), using a graph-based candidate generator "
    "and a LightGBM learning-to-rank model."
)

# ---------------------------------------------------------------------------
# Backend health check
# ---------------------------------------------------------------------------
def check_backend():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return None


health = check_backend()

if health is None:
    st.error(
        "⚠️ Cannot reach the backend API. Make sure it's running:\n\n"
        "`uvicorn src.api.main:app --reload`\n\n"
        f"Expected at {API_BASE_URL}"
    )
    st.stop()

with st.sidebar:
    st.subheader("Backend Status")
    st.success("Connected") if health["status"] == "ok" else st.warning("Degraded")
    st.metric("Ports in graph", health["num_nodes"])
    st.metric("Routes (edges)", health["num_edges"])
    st.write(f"Model loaded: {'✅' if health['model_loaded'] else '❌'}")

# ---------------------------------------------------------------------------
# Port search helper
# ---------------------------------------------------------------------------
def search_ports(query: str):
    if not query or len(query) < 2:
        return []
    try:
        resp = requests.get(f"{API_BASE_URL}/ports", params={"query": query}, timeout=5)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        return resp.json()["results"]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Scenario presets (for quick demo storytelling)
# ---------------------------------------------------------------------------
SCENARIOS = {
    "Custom": None,
    "🔴 Red Sea Crisis: Rotterdam → Singapore": ("rotterdam", "singapore"),
    "🔴 Red Sea Crisis: Hamburg → Mumbai": ("hamburg", "mumbai"),
    "🟢 Unaffected: Los Angeles → Tokyo": ("los angeles", "tokyo ko"),
}

st.subheader("1. Choose a scenario or search manually")
scenario_choice = st.selectbox("Quick demo scenarios", list(SCENARIOS.keys()))

col1, col2 = st.columns(2)

if SCENARIOS[scenario_choice] is not None:
    default_origin, default_dest = SCENARIOS[scenario_choice]
else:
    default_origin, default_dest = "", ""

with col1:
    origin_query = st.text_input("Origin port", value=default_origin, key="origin_input")
    origin_matches = search_ports(origin_query)
    origin_options = {f"{m['port_name']} ({m['country']})": m["port_id"] for m in origin_matches}
    origin_selection = st.selectbox("Select origin match", list(origin_options.keys()) or ["No matches"])

with col2:
    dest_query = st.text_input("Destination port", value=default_dest, key="dest_input")
    dest_matches = search_ports(dest_query)
    dest_options = {f"{m['port_name']} ({m['country']})": m["port_id"] for m in dest_matches}
    dest_selection = st.selectbox("Select destination match", list(dest_options.keys()) or ["No matches"])

top_n = st.slider("Number of ranked routes to show", min_value=1, max_value=10, value=5)

st.subheader("2. Get recommendations")
run_button = st.button("🔍 Recommend Routes", type="primary")

# ---------------------------------------------------------------------------
# Run recommendation
# ---------------------------------------------------------------------------
if run_button:
    if not origin_options or not dest_options:
        st.error("Please select a valid origin and destination from the dropdowns.")
        st.stop()

    origin_id = origin_options[origin_selection]
    dest_id = dest_options[dest_selection]

    with st.spinner("Generating and ranking candidate routes..."):
        try:
            resp = requests.post(
                f"{API_BASE_URL}/recommend",
                json={
                    "origin_port_id": origin_id,
                    "destination_port_id": dest_id,
                    "top_n": top_n,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.HTTPError as e:
            st.error(f"API error: {e.response.json().get('detail', str(e))}")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    st.success(f"Route: **{data['origin']}** → **{data['destination']}**")

    recommendations = data["recommendations"]

    # --- Summary table ---
    st.subheader("Ranked Recommendations")
    table_rows = []
    for i, r in enumerate(recommendations, 1):
        table_rows.append({
            "Rank": i,
            "Score": r["score"],
            "Distance (nm)": r["total_distance_nm"],
            "Hops": r["num_hops"],
            "Max Disruption Exposure": r["disruption_exposure_max"],
            "Disrupted Ports on Route": r["num_disrupted_ports"],
        })
    st.dataframe(table_rows, use_container_width=True)

    # --- Highlight top pick ---
    top = recommendations[0]
    st.markdown(
        f"**Top recommendation:** {top['route'][0]} → ... → {top['route'][-1]} "
        f"({top['num_hops']} hops, {top['total_distance_nm']:.0f} nm, "
        f"disruption exposure {top['disruption_exposure_max']:.2f})"
    )
    if top["disruption_exposure_max"] > 0:
        st.warning(
            f"⚠️ This route passes through {top['num_disrupted_ports']} disrupted port(s). "
            "It was still ranked #1 because the distance savings outweighed the disruption penalty "
            "under the current utility weighting."
        )
    else:
        st.info("✅ This route fully avoids all currently disrupted ports.")

    # --- Map visualization ---
    st.subheader("Route Map")
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
              "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    for i, r in enumerate(recommendations):
        coords = r["coordinates"]
        lats = [c["latitude"] for c in coords]
        lons = [c["longitude"] for c in coords]
        is_top = (i == 0)

        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines+markers" if is_top else "lines",
            line=dict(width=3 if is_top else 1, color=colors[i % len(colors)]),
            marker=dict(size=5 if is_top else 3, color=colors[i % len(colors)]),
            name=f"Rank {i+1}" + (" (Top pick)" if is_top else ""),
            opacity=1.0 if is_top else 0.4,
        ))

    # Mark disrupted ports across the top route
    disrupted_coords = [c for c in recommendations[0]["coordinates"] if c["disrupted"]]
    if disrupted_coords:
        fig.add_trace(go.Scattergeo(
            lon=[c["longitude"] for c in disrupted_coords],
            lat=[c["latitude"] for c in disrupted_coords],
            mode="markers",
            marker=dict(size=12, color="red", symbol="x"),
            name="Disrupted port",
        ))

    fig.update_geos(
        projection_type="natural earth",
        showland=True, landcolor="rgb(243, 243, 243)",
        showocean=True, oceancolor="rgb(220, 235, 245)",
        showcountries=True,
    )
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0))

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full route port sequence (top pick)"):
        st.write(" → ".join(recommendations[0]["route"]))
