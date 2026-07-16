"""
app.py – Streamlit user interface for the Conference Host City IDSS.

Run with:
    streamlit run src/app.py

The UI lets a conference planner:
  1. Enter attendee origin cities and expected headcounts (editable table)
  2. Restrict the candidate host list (or evaluate all 115 cities)
  3. Toggle whether attendee origin cities may be recommended as hosts
  4. See the ranked recommendation, cost comparison chart, map, and
     per-source fare breakdown — all recomputed live from the shipped model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from optimize import load_artifacts, run_optimizer

# ── Palette (colorblind-safe, validated) ─────────────────────────────────────
BLUE       = "#2a78d6"   # recommended host / candidate hosts
BLUE_DARK  = "#1c5cab"   # emphasis
BLUE_LIGHT = "#9ec5f4"   # non-recommended bars
AQUA       = "#1baf7a"   # attendee origin cities
GRAY_TEXT  = "#52514e"

st.set_page_config(
    page_title="Conference Host City IDSS",
    page_icon="🛬",
    layout="wide",
)


# ── Cached loading / computation ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts …")
def get_artifacts() -> dict:
    return load_artifacts()


@st.cache_data(show_spinner="Scoring candidate host cities …")
def score_hosts(
    sources_key: tuple[tuple[str, int], ...],
    candidates_key: tuple[str, ...] | None,
) -> pd.DataFrame:
    arts = get_artifacts()
    sources = [{"city": c, "attendees": a} for c, a in sources_key]
    candidates = list(candidates_key) if candidates_key is not None else None
    return run_optimizer(sources, arts, candidates=candidates)


try:
    arts = get_artifacts()
except FileNotFoundError as exc:
    st.error(
        "Model artifacts not found. Run the pipeline first:\n\n"
        "```\npython3 src/build.py\npython3 src/model.py\n```\n\n"
        f"Details: {exc}"
    )
    st.stop()

ALL_CITIES = arts["all_cities"]

# Display-only geocode corrections for the map. The DOT source data geocodes
# 34 of the 115 cities to the wrong place (name collisions — e.g. Dallas/Fort
# Worth lands in Pennsylvania, Miami in Minneapolis). The model still uses the
# source geocodes (README limitation #6); these overrides only fix where the
# map draws each city.
DISPLAY_COORD_FIXES: dict[str, tuple[float, float]] = {
    "Atlantic City, NJ":               (39.364, -74.423),
    "Bend/Redmond, OR":                (44.160, -121.240),
    "Boise, ID":                       (43.615, -116.202),
    "Columbia, SC":                    (34.000, -81.035),
    "Dallas/Fort Worth, TX":           (32.897, -97.038),
    "Eugene, OR":                      (44.052, -123.087),
    "Fargo, ND":                       (46.877, -96.790),
    "Fort Myers, FL":                  (26.640, -81.873),
    "Grand Rapids, MI":                (42.963, -85.668),
    "Greenville/Spartanburg, SC":      (34.852, -82.394),
    "Harlingen/San Benito, TX":        (26.191, -97.696),
    "Key West, FL":                    (24.556, -81.780),
    "Lubbock, TX":                     (33.578, -101.855),
    "Medford, OR":                     (42.327, -122.876),
    "Miami, FL (Metropolitan Area)":   (25.762, -80.192),
    "Mission/McAllen/Edinburg, TX":    (26.204, -98.230),
    "Omaha, NE":                       (41.257, -95.935),
    "Pasco/Kennewick/Richland, WA":    (46.230, -119.100),
    "Phoenix, AZ":                     (33.448, -112.074),
    "Portland, ME":                    (43.661, -70.255),
    "Provo, UT":                       (40.234, -111.658),
    "Punta Gorda, FL":                 (26.930, -82.045),
    "Raleigh/Durham, NC":              (35.878, -78.788),
    "Reno, NV":                        (39.530, -119.814),
    "Sanford, FL":                     (28.800, -81.273),
    "Savannah, GA":                    (32.081, -81.091),
    "Tallahassee, FL":                 (30.438, -84.281),
    "Tampa, FL (Metropolitan Area)":   (27.951, -82.457),
    "Tucson, AZ":                      (32.222, -110.975),
    "Tulsa, OK":                       (36.154, -95.993),
    "Valparaiso, FL":                  (30.508, -86.503),
    "West Palm Beach/Palm Beach, FL":  (26.715, -80.054),
    "Wichita, KS":                     (37.687, -97.336),
    "Wilmington, NC":                  (34.226, -77.945),
}

COORDS = arts["coords_df"].set_index("city").copy()
for _city, (_lat, _lon) in DISPLAY_COORD_FIXES.items():
    if _city in COORDS.index:
        COORDS.loc[_city, ["lat", "lon"]] = (_lat, _lon)


# ── Sidebar: user inputs ─────────────────────────────────────────────────────

st.sidebar.title("Planner inputs")

st.sidebar.subheader("1 · Attendee origins")
st.sidebar.caption("Where attendees fly from, and how many from each city.")

if "origins" not in st.session_state:
    st.session_state.origins = {
        "Chicago, IL": 30,
        "Dallas/Fort Worth, TX": 50,
    }


def _set_attendees(city: str) -> None:
    st.session_state.origins[city] = int(st.session_state[f"att_{city}"])


def _remove_origin(city: str) -> None:
    st.session_state.origins.pop(city, None)


# One card per origin. Two fixed lines — city + remove button, then
# stepper + "attendees" — so nothing competes for width and the layout
# stays aligned however the sidebar is resized.
for _city in list(st.session_state.origins):
    with st.sidebar.container(border=True):
        with st.container(
            horizontal=True, vertical_alignment="center", gap="small"
        ):
            st.markdown(
                f'<span style="font-size:0.9rem; font-weight:600;">'
                f"{_city}</span>",
                unsafe_allow_html=True,
                width="stretch",
            )
            st.button(
                "✕", key=f"rm_{_city}",
                on_click=_remove_origin, args=(_city,),
                help=f"Remove {_city}",
                width="content",
            )
        with st.container(
            horizontal=True, vertical_alignment="center", gap="small"
        ):
            st.number_input(
                f"Attendees from {_city}",
                min_value=1, step=5,
                value=int(st.session_state.origins[_city]),
                key=f"att_{_city}",
                on_change=_set_attendees, args=(_city,),
                label_visibility="collapsed",
                width=110,
            )
            st.markdown(
                '<span style="white-space:nowrap; font-size:0.8rem; '
                'opacity:0.6;">attendees</span>',
                unsafe_allow_html=True,
                width="content",
            )

if st.session_state.origins:
    st.sidebar.caption(
        f"**{len(st.session_state.origins)} origin"
        f"{'s' if len(st.session_state.origins) != 1 else ''} · "
        f"{sum(st.session_state.origins.values())} attendees**"
    )

# Add-a-city form: searchable dropdown + headcount, one click to add
_addable = [c for c in ALL_CITIES if c not in st.session_state.origins]
with st.sidebar.form("add_origin", clear_on_submit=True):
    new_city = st.selectbox(
        "Add an origin city",
        options=_addable,
        index=None,
        placeholder="Type to search 115 cities…",
    )
    new_att = st.number_input(
        "Attendees from there", min_value=1, value=25, step=5
    )
    if st.form_submit_button("＋ Add origin", type="primary", width="stretch"):
        if new_city is None:
            st.warning("Pick a city first.")
        else:
            st.session_state.origins[new_city] = int(new_att)
            st.rerun()

st.sidebar.subheader("2 · Candidate host cities")
candidate_pick = st.sidebar.multiselect(
    "Restrict candidates (empty = all 115 cities)",
    options=ALL_CITIES,
    default=[],
)

exclude_sources = st.sidebar.toggle(
    "Exclude origin cities from hosting",
    value=True,
    help=(
        "Origin cities pay $0 airfare for their own attendees, so with this "
        "off the optimizer almost always recommends one of them. Turn this on "
        "to force a neutral host city."
    ),
)

st.sidebar.subheader("3 · Fare data vintage")
st.sidebar.selectbox(
    "Quarter",
    options=["2025 Q4 (latest DOT release)"],
    help=(
        "Fares come from the DOT Consumer Airfare Report, released quarterly. "
        "Re-running the pipeline after each release refreshes this."
    ),
)

top_n = st.sidebar.slider("Cities shown in comparison", 3, 20, 8)


# ── Validate & resolve inputs ────────────────────────────────────────────────

st.title("Conference Host City Decision Support")
st.caption(
    "Ranks candidate U.S. host cities by total round-trip airfare for your "
    "attendees, using DOT observed fares where available and a LightGBM fare "
    "model elsewhere."
)

# Cities come from the canonical dropdown, so no fuzzy matching is needed
resolved: dict[str, int] = {
    c: int(a) for c, a in st.session_state.origins.items() if int(a) > 0
}

if not resolved:
    st.info("Add at least one attendee origin city in the sidebar to begin.")
    st.stop()

source_cities = list(resolved)
total_attendees = sum(resolved.values())

# Build candidate list, applying the exclusion toggle
candidates = list(candidate_pick) if candidate_pick else list(ALL_CITIES)
excluded_hosts = [c for c in candidates if c in resolved] if exclude_sources else []
if exclude_sources:
    candidates = [c for c in candidates if c not in resolved]

if not candidates:
    st.error(
        "No candidate host cities remain after excluding the origin cities. "
        "Add more candidates or turn the exclusion toggle off."
    )
    st.stop()

sources_key = tuple(sorted(resolved.items()))
candidates_key = tuple(candidates)
result = score_hosts(sources_key, candidates_key)

best = result.iloc[0]
runner_up = result.iloc[1] if len(result) > 1 else None


# ── Recommendation banner ────────────────────────────────────────────────────

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Recommended host", best["host"].split(",")[0])
c2.metric("Total travel cost", f"${best['total_cost']:,.0f}")
c3.metric("Cost per attendee", f"${best['cost_per_attendee']:,.0f}")
if runner_up is not None:
    savings = runner_up["total_cost"] - best["total_cost"]
    c4.metric(
        "Savings vs. runner-up",
        f"${savings:,.0f}",
        f"{runner_up['host'].split(',')[0]} is next best",
        delta_color="off",
    )

detail_bits = [f"{a} attendees from {c}" for c, a in resolved.items()]
st.markdown(
    f"For **{total_attendees} attendees** ({'; '.join(detail_bits)}), "
    f"**{best['host']}** minimizes total round-trip airfare across the "
    f"{len(candidates)} candidate host cities evaluated."
)
if excluded_hosts:
    st.caption(
        "Excluded from hosting (origin cities): " + ", ".join(excluded_hosts)
    )

imputed_share = float(best["imputed_attendee_share"])
if imputed_share > 0.5:
    st.warning(
        f"{imputed_share:.0%} of the recommended city's cost comes from "
        "model-predicted fares (routes not in the DOT top-1,000 markets). "
        "Treat this ranking as an estimate and verify fares before booking."
    )


# ── Tabs: chart / map / table / breakdown ────────────────────────────────────

tab_chart, tab_map, tab_table, tab_detail = st.tabs(
    ["Cost comparison", "Map", "Full ranking", "Fare breakdown"]
)

top = result.head(top_n).copy()

with tab_chart:
    top_sorted = top.sort_values("total_cost", ascending=False)
    colors = [
        BLUE if h == best["host"] else BLUE_LIGHT for h in top_sorted["host"]
    ]
    fig = go.Figure(
        go.Bar(
            x=top_sorted["total_cost"],
            y=top_sorted["host"],
            orientation="h",
            marker=dict(color=colors, cornerradius=4),
            text=[f"${v:,.0f}" for v in top_sorted["total_cost"]],
            textposition="outside",
            hovertemplate=(
                "<b>%{y}</b><br>Total cost: $%{x:,.0f}<br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Total round-trip travel cost — top {len(top)} candidate hosts",
        xaxis_title="Total cost (USD)",
        xaxis=dict(
            tickformat="$,.0f", showgrid=True,
            range=[0, float(top_sorted["total_cost"].max()) * 1.15],
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=90 + 42 * len(top),
        margin=dict(l=10, r=80, t=50, b=40),
    )
    fig.update_yaxes(autorange=True)  # keep sorted order (cheapest at top)
    st.plotly_chart(
        fig, width="stretch", config={"displayModeBar": False}
    )
    st.caption(
        "The highlighted bar is the recommended host. Bar length = airfare "
        "the whole group pays; picking a longer bar costs the difference."
    )

with tab_map:
    fig = go.Figure()

    # Candidate hosts (top N), sized by rank, blue
    host_rows = top.merge(
        COORDS[["lat", "lon"]], left_on="host", right_index=True, how="left"
    )
    fig.add_trace(
        go.Scattergeo(
            lon=host_rows["lon"],
            lat=host_rows["lat"],
            text=[
                f"#{r} {h}<br>Total: ${t:,.0f}"
                for r, (h, t) in enumerate(
                    zip(host_rows["host"], host_rows["total_cost"]), start=1
                )
            ],
            hoverinfo="text",
            mode="markers",
            name=f"Top {len(top)} candidate hosts",
            marker=dict(size=11, color=BLUE, line=dict(width=1, color="white")),
        )
    )

    # Recommended host highlighted
    b = COORDS.loc[best["host"]]
    fig.add_trace(
        go.Scattergeo(
            lon=[b["lon"]], lat=[b["lat"]],
            text=[f"RECOMMENDED: {best['host']}<br>${best['total_cost']:,.0f}"],
            hoverinfo="text",
            mode="markers",
            name="Recommended host",
            marker=dict(
                size=20, color=BLUE_DARK, symbol="star",
                line=dict(width=1, color="white"),
            ),
        )
    )

    # Origin cities, aqua, sized by attendees
    src_lon = [COORDS.loc[c, "lon"] for c in source_cities]
    src_lat = [COORDS.loc[c, "lat"] for c in source_cities]
    max_att = max(resolved.values())
    fig.add_trace(
        go.Scattergeo(
            lon=src_lon, lat=src_lat,
            text=[f"Origin: {c}<br>{resolved[c]} attendees" for c in source_cities],
            hoverinfo="text",
            mode="markers",
            name="Attendee origins",
            marker=dict(
                size=[10 + 14 * resolved[c] / max_att for c in source_cities],
                color=AQUA, symbol="diamond",
                line=dict(width=1, color="white"),
            ),
        )
    )

    # Flight lines origin → recommended host
    for c in source_cities:
        if c == best["host"]:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=[COORDS.loc[c, "lon"], b["lon"]],
                lat=[COORDS.loc[c, "lat"], b["lat"]],
                mode="lines",
                line=dict(width=2.5, color=BLUE_DARK),
                opacity=0.8,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        geo=dict(
            scope="usa",
            bgcolor="#eaf1f8",       
            landcolor="#dfe1e5",      
            lakecolor="#eaf1f8",
            showland=True,
            showlakes=True,
            showsubunits=True,
            subunitcolor="#ffffff",  
            subunitwidth=1.0,
            showcoastlines=True,
            coastlinecolor="#9aa0a6",
            showframe=False,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02, x=0.02,
            bgcolor="rgba(255,255,255,0.75)", font=dict(color="#333333"),
        ),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption(
        "Diamonds are attendee origins (sized by headcount); circles are the "
        "top candidate hosts; the star is the recommendation. Lines show the "
        "flights your attendees would take. Map positions use corrected "
        "geocodes; the fare model itself still uses the DOT source geocodes "
        "(see README limitation #6)."
    )

with tab_table:
    show = result[
        ["host", "total_cost", "cost_per_attendee",
         "n_imputed_legs", "imputed_attendee_share"]
    ].rename(
        columns={
            "host": "Host city",
            "total_cost": "Total cost",
            "cost_per_attendee": "Cost / attendee",
            "n_imputed_legs": "Predicted legs",
            "imputed_attendee_share": "Predicted-fare share",
        }
    )
    st.dataframe(
        show,
        width="stretch",
        height=480,
        column_config={
            "Total cost": st.column_config.NumberColumn(format="$%,.0f"),
            "Cost / attendee": st.column_config.NumberColumn(format="$%,.0f"),
            "Predicted-fare share": st.column_config.ProgressColumn(
                format="percent", min_value=0.0, max_value=1.0,
                help="Share of cost based on model predictions rather than "
                     "observed DOT fares. Higher = less certain.",
            ),
        },
    )
    st.caption(
        "‘Predicted legs’ counts origin→host routes priced by the LightGBM "
        "model because they are outside the DOT top-1,000 observed markets."
    )

with tab_detail:
    pick = st.selectbox(
        "Inspect a candidate host",
        options=list(result["host"]),
        index=0,
    )
    row = result[result["host"] == pick].iloc[0]
    bd = pd.DataFrame(row["breakdown"]).sort_values("leg_cost", ascending=False)
    bd["fare_source"] = bd["imputed"].map(
        {True: "Model prediction", False: "DOT observed"}
    )
    bd["round_trip"] = 2 * bd["fare_ow"]
    st.markdown(
        f"**{pick}** — total **\\${row['total_cost']:,.0f}** "
        f"(\\${row['cost_per_attendee']:,.0f}/attendee)"
        + (
            f", **\\${row['total_cost'] - best['total_cost']:,.0f} more** than "
            f"recommended {best['host']}"
            if pick != best["host"] else " — recommended host"
        )
    )
    st.dataframe(
        bd[["source", "attendees", "fare_ow", "round_trip", "leg_cost",
            "fare_source"]].rename(
            columns={
                "source": "Origin",
                "attendees": "Attendees",
                "fare_ow": "One-way fare",
                "round_trip": "Round trip",
                "leg_cost": "Group cost",
                "fare_source": "Fare source",
            }
        ),
        width="stretch",
        hide_index=True,
        column_config={
            "One-way fare": st.column_config.NumberColumn(format="$%,.2f"),
            "Round trip": st.column_config.NumberColumn(format="$%,.2f"),
            "Group cost": st.column_config.NumberColumn(format="$%,.0f"),
        },
    )

st.divider()
st.caption(
    "Fares are DOT quarterly average one-way market fares (all carriers), not "
    "bookable prices. Round trip is approximated as 2× one-way. Routes outside "
    "the top-1,000 DOT markets use LightGBM predictions (CV MAE ≈ $33)."
)
