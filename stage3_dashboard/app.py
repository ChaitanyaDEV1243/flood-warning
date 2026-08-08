# app.py
import streamlit as st
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Periyar Flood Early Warning System", layout="wide")

# ---------------------------------------------------------------------------
# LOAD DATA (all six files must sit in the same folder as this script)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    zones = pd.read_csv("zones_with_risk_clean.csv")
    with open("allocation_comparison.json") as f:
        allocation = json.load(f)
    district_summary = pd.read_csv("district_validation_summary.csv")
    rainfall_val = pd.read_csv("rainfall_warning_validation.csv")
    with open("validation_summary.json") as f:
        validation = json.load(f)
    return zones, allocation, district_summary, rainfall_val, validation

try:
    zones, allocation, district_summary, rainfall_val, validation = load_data()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e}. Make sure all six files from the Colab export "
        "are in the same folder as app.py."
    )
    st.stop()

# ---------------------------------------------------------------------------
# SIDEBAR NAV
# ---------------------------------------------------------------------------
st.sidebar.title("🌊 Periyar Flood EWS")
page = st.sidebar.radio(
    "View",
    ["Risk Heatmap", "Allocation Overlay", "Timeline Replay", "2018 Validation", "SMS Alert Mock"]
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Stage 1 (GAT): MAE 0.053, R² 0.651\n\n"
    "Stage 2: Greedy/LP baseline vs RL (PPO)\n\n"
    "Validated against Kerala 2018 floods"
)

# ---------------------------------------------------------------------------
# PAGE 1: RISK HEATMAP
# ---------------------------------------------------------------------------
if page == "Risk Heatmap":
    st.title("Flood Risk Heatmap — 11 Periyar Basin Zones")
    st.caption("Risk scores from Person A's GAT model (0 = low risk, 1 = high risk)")

    col1, col2 = st.columns([3, 1])

    with col1:
        fig = px.scatter_mapbox(
            zones,
            lat="lat", lon="lon",
            color="risk_score",
            size="Population Density",
            hover_name="zone_name",
            hover_data={
                "risk_score": ":.3f",
                "district": True,
                "Water Level (m)": ":.2f",
                "Rainfall (mm)": ":.1f",
                "lat": False, "lon": False
            },
            color_continuous_scale="YlOrRd",
            size_max=30,
            zoom=8,
            mapbox_style="carto-positron",
            title="Zone-level flood risk"
        )
        fig.update_layout(height=600, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Ranked by risk")
        ranked = zones[["zone_name", "district", "risk_score"]].sort_values(
            "risk_score", ascending=False
        ).reset_index(drop=True)
        ranked.index += 1
        st.dataframe(ranked, use_container_width=True, height=560)

    st.markdown("---")
    st.subheader("Model performance (Stage 1)")
    m1, m2, m3 = st.columns(3)
    m1.metric("Baseline MLP", "MAE 0.085", "R² -0.135")
    m2.metric("GCN", "MAE 0.058", "R² 0.571")
    m3.metric("GAT (final)", "MAE 0.053", "R² 0.651", delta_color="normal")

# ---------------------------------------------------------------------------
# PAGE 2: ALLOCATION OVERLAY
# ---------------------------------------------------------------------------
elif page == "Allocation Overlay":
    st.title("Rescue Resource Allocation — Baseline vs RL")

    baseline_zones = set(allocation["baseline"]["allocation_order"])
    rl_zones = set(allocation["rl"]["allocation_order"])
    swapped_out = allocation["diff"]["swapped_out"]
    swapped_in = allocation["diff"]["swapped_in"]

    def get_group(zone_name):
        in_base = zone_name in baseline_zones
        in_rl = zone_name in rl_zones
        if in_base and in_rl:
            return "Selected by both"
        elif in_base:
            return "Baseline only (dropped by RL)"
        elif in_rl:
            return "RL only (added by RL)"
        else:
            return "Not selected"

    zones_plot = zones.copy()
    zones_plot["allocation_group"] = zones_plot["zone_name"].apply(get_group)

    color_map = {
        "Selected by both": "#2ca02c",
        "Baseline only (dropped by RL)": "#d62728",
        "RL only (added by RL)": "#1f77b4",
        "Not selected": "#cccccc"
    }

    fig = px.scatter_mapbox(
        zones_plot,
        lat="lat", lon="lon",
        color="allocation_group",
        hover_name="zone_name",
        hover_data={"risk_score": ":.3f", "lat": False, "lon": False},
        color_discrete_map=color_map,
        zoom=8,
        mapbox_style="carto-positron",
        height=550
    )
    fig.update_traces(marker=dict(size=18))
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Greedy / LP Baseline")
        st.caption(allocation["baseline"]["method"])
        for z in allocation["baseline"]["allocation_order"]:
            st.write(f"✅ {z}")
        st.metric("Coverage", f"{allocation['baseline']['coverage_value']:,.1f}")

    with c2:
        st.subheader("RL Agent (PPO)")
        st.caption(allocation["rl"]["method"])
        for z in allocation["rl"]["allocation_order"]:
            marker = "🆕" if z in swapped_in else "✅"
            st.write(f"{marker} {z}")

    st.info(
        f"**What changed:** RL dropped **{', '.join(swapped_out)}** "
        f"(picked by the static greedy/LP baseline) and instead selected "
        f"**{', '.join(swapped_in)}** — because RL accounts for how risk "
        f"evolves over time, not just a single snapshot."
    )

# ---------------------------------------------------------------------------
# PAGE 3: TIMELINE REPLAY (illustrative — see caption)
# ---------------------------------------------------------------------------
elif page == "Timeline Replay":
    st.title("Risk Build-Up Replay")
    st.warning(
        "⚠️ **Illustrative replay.** Our Stage 1 model outputs a single risk "
        "snapshot per zone, not a time series. This animation interpolates "
        "from a calm baseline toward the real predicted risk scores, "
        "purely to visualize how a flood event might unfold. It is not "
        "real temporal model output."
    )

    frame = st.slider("Simulated hours into event", 0, 100, 0, step=5)
    t = frame / 100.0

    zones_t = zones.copy()
    # start near-zero risk, interpolate linearly to the real predicted risk_score
    zones_t["display_risk"] = zones_t["risk_score"] * t

    fig = px.scatter_mapbox(
        zones_t,
        lat="lat", lon="lon",
        color="display_risk",
        size="Population Density",
        hover_name="zone_name",
        hover_data={"display_risk": ":.3f", "lat": False, "lon": False},
        color_continuous_scale="YlOrRd",
        range_color=[0, zones["risk_score"].max()],
        size_max=30,
        zoom=8,
        mapbox_style="carto-positron",
        height=550
    )
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Frame: {frame}% of the way to final predicted risk state")

# ---------------------------------------------------------------------------
# PAGE 4: 2018 VALIDATION
# ---------------------------------------------------------------------------
elif page == "2018 Validation":
    st.title("Validation Against the 2018 Kerala Floods")

    st.subheader("Zone risk vs district-level outcomes")
    fig1 = px.bar(
        district_summary,
        x="district", y="avg_risk_score",
        hover_data=["fatalities", "no_of_landslides", "full_damaged_houses"],
        color="district",
        title="Average predicted zone risk, by district"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.dataframe(district_summary, use_container_width=True)

    st.markdown("---")
    st.subheader("Rainfall warning-level accuracy (Idukki + Ernakulam, 2018)")
    acc = validation["rainfall_warning_accuracy"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Exact match rate", f"{acc['exact_match_rate']:.1%}")
    m2.metric("Mean warning-level error", f"{acc['mean_absolute_level_error']:.2f}", "0=exact, 3=max")
    m3.metric("Observations", acc["n_observations"])

    per_district = pd.DataFrame(acc["per_district"]).T.reset_index().rename(columns={"index": "district"})
    fig2 = px.bar(per_district, x="district", y="exact_match", title="Exact-match rate by district", range_y=[0, 1])
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("⚠️ Honest limitations of this validation")
    for lim in validation["limitations"]:
        st.write(f"- {lim}")

# ---------------------------------------------------------------------------
# PAGE 5: MOCKED SMS ALERT (placeholder — wired up in the next step)
# ---------------------------------------------------------------------------
elif page == "SMS Alert Mock":
    st.title("SMS Alert Flow (Mock)")
    st.info("Twilio sandbox integration goes here — next step.")

    threshold = st.slider("Alert threshold (risk score)", 0.0, 1.0, 0.40, 0.01)
    high_risk = zones[zones["risk_score"] >= threshold].sort_values("risk_score", ascending=False)

    st.write(f"**{len(high_risk)} zone(s)** would trigger an alert at threshold {threshold:.2f}:")
    st.dataframe(high_risk[["zone_name", "district", "risk_score"]], use_container_width=True)

    if st.button("Send mock alerts"):
        for _, row in high_risk.iterrows():
            st.success(f"📱 [MOCK] SMS sent for {row['zone_name']} — risk {row['risk_score']:.3f}")