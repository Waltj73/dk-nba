import streamlit as st
import pandas as pd
import requests
import pulp
import io
from nba_api.stats.endpoints import leaguedashplayerstats

st.set_page_config(
    page_title="Autonomous DK NBA Optimizer",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 Automated DraftKings NBA DFS Engine")
st.caption("Auto-fetch Live Slate -> Auto-project with Official NBA Stats -> Integer Linear Optimization")

# --- 1. DRAFTKINGS API FETCHERS ---
@st.cache_data(ttl=600)
def fetch_dk_nba_contests():
    """Fetch active NBA contests and draft group IDs directly from DraftKings."""
    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        contests = data.get("Contests", [])
        
        # Group by DraftGroupId (slate)
        slates = {}
        for c in contests:
            d_id = c.get("dg")
            name = c.get("n", "NBA Slate")
            if d_id and d_id not in slates:
                slates[d_id] = f"Slate {d_id}: {name[:45]}"
        return slates
    except Exception as e:
        st.warning(f"DraftKings Lobby API Notice: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_dk_players_by_draftgroup(draft_group_id):
    """Pulls players, salaries, positions, and games directly for a Draft Group."""
    url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables?format=json"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    data = res.json()
    draftables = data.get("draftables", [])
    
    records = []
    for p in draftables:
        # Filter out disabled / inactive players
        if p.get("isDisabled", False):
            continue
            
        name = p.get("displayName")
        salary = p.get("salary", 0)
        position = p.get("position", "")
        team = p.get("teamAbbreviation", "")
        
        # Extract competition info
        comp = p.get("competition", {})
        game_desc = comp.get("name", "NBA Game")
        
        records.append({
            "PlayerId": p.get("playerId"),
            "Name": name,
            "Position": position,
            "Salary": salary,
            "TeamAbbrev": team,
            "Game Info": game_desc
        })
        
    df = pd.DataFrame(records).drop_duplicates(subset=["Name", "Position"])
    return df

# --- 2. NBA STATS & PROJECTION ENGINE ---
@st.cache_data(ttl=3600)
def get_season_nba_projections():
    """Fetch official per-game stats via nba_api and compute DK fantasy point averages."""
    try:
        # Pull per-game league dashboard
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='PerGame',
            season='2025-26', # current season context
            season_type_all_star='Regular Season'
        )
        stats_df = dash.get_data_frames()[0]
        
        # DraftKings scoring calculation
        # PTS(1.0) + 3PM(0.5) + REB(1.25) + AST(1.5) + STL(2.0) + BLK(2.0) - TO(0.5)
        stats_df["Base_DK_FPTS"] = (
            stats_df["PTS"] * 1.0 +
            stats_df["FG3M"] * 0.5 +
            stats_df["REB"] * 1.25 +
            stats_df["AST"] * 1.5 +
            stats_df["STL"] * 2.0 +
            stats_df["BLK"] * 2.0 -
            stats_df["TOV"] * 0.5
        )
        
        # Approximate Double-Double bonus (+1.5) for top contributors
        stats_df["DoubleDouble_Prob"] = (
            ((stats_df["PTS"] >= 10).astype(int) + 
             (stats_df["REB"] >= 10).astype(int) + 
             (stats_df["AST"] >= 10).astype(int)) >= 2
        ).astype(int) * 1.5
        
        stats_df["DK_Projected_FPTS"] = (stats_df["Base_DK_FPTS"] + stats_df["DoubleDouble_Prob"]).round(2)
        
        return stats_df[["PLAYER_NAME", "MIN", "DK_Projected_FPTS", "PTS", "REB", "AST"]]
    except Exception as e:
        st.error(f"Error fetching stats from nba_api: {e}")
        return pd.DataFrame()

# --- 3. LINEUP OPTIMIZER (PuLP) ---
def optimize_roster(df_pool, locks, excludes, cap=50000, min_games=2):
    active = df_pool[~df_pool["Name"].isin(excludes)].copy().reset_index(drop=True)
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    def is_eligible(pos_str, slot):
        positions = [p.strip() for p in pos_str.split("/")]
        if slot == "UTIL":
            return True
        if slot == "G":
            return any(p in ["PG", "SG"] for p in positions)
        if slot == "F":
            return any(p in ["SF", "PF"] for p in positions)
        return slot in positions

    prob = pulp.LpProblem("DK_NBA_Optimization", pulp.LpMaximize)
    indices = list(active.index)

    # Decision variables
    x = {}
    for i in indices:
        for s in slots:
            if is_eligible(active.loc[i, "Position"], s):
                x[i, s] = pulp.LpVariable(f"x_{i}_{s}", cat="Binary")
            else:
                x[i, s] = 0

    # Maximize total projection
    prob += pulp.lpSum(active.loc[i, "Projected_FPTS"] * pulp.lpSum(x[i, s] for s in slots) for i in indices)

    # Exactly 1 player per slot
    for s in slots:
        prob += pulp.lpSum(x[i, s] for i in indices) == 1

    # Max 1 selection per player
    for i in indices:
        prob += pulp.lpSum(x[i, s] for s in slots) <= 1

    # Salary cap
    prob += pulp.lpSum(active.loc[i, "Salary"] * pulp.lpSum(x[i, s] for s in slots) for i in indices) <= cap

    # Locked players
    for lock in locks:
        lock_idx = active[active["Name"] == lock].index
        if len(lock_idx) > 0:
            prob += pulp.lpSum(x[lock_idx[0], s] for s in slots) == 1

    # Unique games constraint
    games = active["Game Info"].unique()
    game_vars = {g: pulp.LpVariable(f"g_{idx}", cat="Binary") for idx, g in enumerate(games)}
    for g in games:
        g_indices = active[active["Game Info"] == g].index
        prob += pulp.lpSum(x[i, s] for i in g_indices for s in slots) >= game_vars[g]
        prob += pulp.lpSum(x[i, s] for i in g_indices for s in slots) <= 8 * game_vars[g]
    prob += pulp.lpSum(game_vars[g] for g in games) >= min_games

    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None

    roster = []
    for s in slots:
        for i in indices:
            val = pulp.value(x[i, s])
            if val is not None and val > 0.5:
                row = active.loc[i].to_dict()
                row["Roster Slot"] = s
                roster.append(row)
    return pd.DataFrame(roster)

# --- 4. SIDEBAR & WORKFLOW ---
st.sidebar.header("Slate Selector")
available_slates = fetch_dk_nba_contests()

if available_slates:
    selected_dg = st.sidebar.selectbox("Choose Active Slate", options=list(available_slates.keys()), format_func=lambda x: available_slates[x])
else:
    # Fallback to test/custom draft group if off-hours
    selected_dg = st.sidebar.text_input("Enter DraftGroup ID Manually", value="46589")

salary_cap = st.sidebar.number_input("Salary Cap", value=50000, step=500)

# Pull player slate
with st.spinner("Fetching live DraftKings player pool..."):
    slate_df = fetch_dk_players_by_draftgroup(selected_dg)

if slate_df.empty:
    st.error("No player data returned for this Draft Group. Please select an active slate.")
    st.stop()

# Merge with NBA API statistics
with st.spinner("Calculating fantasy scoring formulas via nba_api..."):
    nba_projections = get_season_nba_projections()
    if not nba_projections.empty:
        merged_df = pd.merge(slate_df, nba_projections, left_on="Name", right_on="PLAYER_NAME", how="left")
        merged_df["Projected_FPTS"] = merged_df["DK_Projected_FPTS"].fillna(merged_df["Salary"] / 250.0).round(2)
        merged_df["MIN"] = merged_df["MIN"].fillna(0.0).round(1)
    else:
        # Fallback baseline formula
        merged_df = slate_df.copy()
        merged_df["Projected_FPTS"] = (merged_df["Salary"] / 220.0).round(2)
        merged_df["MIN"] = 24.0

merged_df["Value (x/1k)"] = (merged_df["Projected_FPTS"] / (merged_df["Salary"] / 1000)).round(2)

# Multi-select rules
st.sidebar.subheader("Player Constraints")
locked = st.sidebar.multiselect("Lock Players", options=merged_df["Name"].unique())
faded = st.sidebar.multiselect("Exclude Players", options=[p for p in merged_df["Name"].unique() if p not in locked])

# --- 5. TABS INTERFACE ---
tab1, tab2 = st.tabs(["⚡ Auto-Generated Optimal Lineup", "📋 Full Automated Pool"])

with tab1:
    if st.button("Generate Mathematically Optimal Lineup", type="primary"):
        with st.spinner("Running PuLP integer solver..."):
            best_lineup = optimize_roster(merged_df, locked, faded, cap=salary_cap)
            
        if best_lineup is None or best_lineup.empty:
            st.error("Infeasible setup. Please loosen your player locks.")
        else:
            tot_pts = best_lineup["Projected_FPTS"].sum().round(2)
            tot_sal = best_lineup["Salary"].sum()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Projected FPTS", f"{tot_pts} pts")
            c2.metric("Total Salary", f"${tot_sal:,}")
            c3.metric("Remaining Cap", f"${salary_cap - tot_sal:,}")
            c4.metric("Avg / Player", f"${int(tot_sal / 8):,}")

            # Sort by DK slot order
            order = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4, "G": 5, "F": 6, "UTIL": 7}
            best_lineup["Sort"] = best_lineup["Roster Slot"].map(order)
            best_lineup = best_lineup.sort_values(by="Sort").drop(columns=["Sort"])

            st.dataframe(
                best_lineup[["Roster Slot", "Name", "Position", "TeamAbbrev", "Salary", "MIN", "Projected_FPTS", "Value (x/1k)", "Game Info"]],
                use_container_width=True,
                hide_index=True
            )

            # Upload-ready DK CSV export
            dk_export = pd.DataFrame([{
                "PG": best_lineup[best_lineup["Roster Slot"] == "PG"]["Name"].values[0],
                "SG": best_lineup[best_lineup["Roster Slot"] == "SG"]["Name"].values[0],
                "SF": best_lineup[best_lineup["Roster Slot"] == "SF"]["Name"].values[0],
                "PF": best_lineup[best_lineup["Roster Slot"] == "PF"]["Name"].values[0],
                "C": best_lineup[best_lineup["Roster Slot"] == "C"]["Name"].values[0],
                "G": best_lineup[best_lineup["Roster Slot"] == "G"]["Name"].values[0],
                "F": best_lineup[best_lineup["Roster Slot"] == "F"]["Name"].values[0],
                "UTIL": best_lineup[best_lineup["Roster Slot"] == "UTIL"]["Name"].values[0],
            }])

            buf = io.StringIO()
            dk_export.to_csv(buf, index=False)
            st.download_button(
                "📥 Download Ready-to-Upload DK CSV",
                data=buf.getvalue(),
                file_name="dk_lineup_auto.csv",
                mime="text/csv"
            )

with tab2:
    st.subheader("Live Slate Pool with Calculated Projections")
    st.dataframe(
        merged_df[["Name", "Position", "TeamAbbrev", "Salary", "MIN", "Projected_FPTS", "Value (x/1k)", "Game Info"]].sort_values(by="Projected_FPTS", ascending=False),
        use_container_width=True,
        hide_index=True
    )
