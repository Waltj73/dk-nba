import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
import io
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

st.set_page_config(
    page_title="Autonomous NBA DFS 5-Lineup Engine",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 Autonomous DraftKings NBA DFS Multi-Lineup Engine")
st.caption("Live Slate Ingestion • Algorithmic Pace/DvP Adjustments • Usage Reallocation • 5 Diverse GPP Rosters")

# --- 1. LIVE DRAFTKINGS LOBBY INGESTION ---
@st.cache_data(ttl=600)
def fetch_dk_nba_contests():
    """Scans DraftKings lobby for active NBA slates (DraftGroups)."""
    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        contests = data.get("Contests", [])
        slates = {}
        for c in contests:
            d_id = c.get("dg")
            name = c.get("n", "NBA Slate")
            if d_id and d_id not in slates:
                slates[d_id] = f"Slate {d_id}: {name[:45]}"
        return slates
    except Exception as e:
        st.warning(f"DraftKings Lobby Fetch Note: {e}")
        return {}

@st.cache_data(ttl=600)
def fetch_dk_players_by_draftgroup(draft_group_id):
    """Pulls full active player pool, positions, salaries, and game info for slate."""
    url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables?format=json"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers, timeout=10)
    data = res.json()
    draftables = data.get("draftables", [])
    
    records = []
    for p in draftables:
        if p.get("isDisabled", False):
            continue
        comp = p.get("competition", {})
        game_desc = comp.get("name", "NBA Game")
        records.append({
            "PlayerId": p.get("playerId"),
            "Name": p.get("displayName"),
            "Position": p.get("position", ""),
            "Salary": int(p.get("salary", 0)),
            "TeamAbbrev": p.get("teamAbbreviation", ""),
            "Game Info": game_desc
        })
    df = pd.DataFrame(records).drop_duplicates(subset=["Name", "Position"])
    return df

# --- 2. NBA STATS & ADVANCED METRICS ENGINE ---
@st.cache_data(ttl=3600)
def fetch_nba_stats():
    """Pulls player averages & team pace from official NBA stats."""
    try:
        # Player Stats (Current Season)
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='PerGame',
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        p_df = dash.get_data_frames()[0]

        # Exact DraftKings Scoring: PTS(1) + 3PM(0.5) + REB(1.25) + AST(1.5) + STL(2) + BLK(2) - TO(0.5)
        p_df["Base_DK_FPTS"] = (
            p_df["PTS"] * 1.0 +
            p_df["FG3M"] * 0.5 +
            p_df["REB"] * 1.25 +
            p_df["AST"] * 1.5 +
            p_df["STL"] * 2.0 +
            p_df["BLK"] * 2.0 -
            p_df["TOV"] * 0.5
        )
        
        # Double-Double / Triple-Double Likelihood Bonus
        p_df["DD_Bonus"] = (
            ((p_df["PTS"] >= 10).astype(int) + 
             (p_df["REB"] >= 10).astype(int) + 
             (p_df["AST"] >= 10).astype(int)) >= 2
        ).astype(int) * 1.5

        p_df["Season_FPTS"] = (p_df["Base_DK_FPTS"] + p_df["DD_Bonus"]).round(2)
        player_clean = p_df[["PLAYER_NAME", "MIN", "Season_FPTS", "PTS", "REB", "AST"]].copy()

        # Team Pace & Def Rating
        t_dash = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        t_df = t_dash.get_data_frames()[0]
        team_map = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
            'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        t_df["TeamAbbrev"] = t_df["TEAM_NAME"].map(team_map)
        team_clean = t_df[["TeamAbbrev", "PACE", "DEF_RATING"]].dropna()

        return player_clean, team_clean
    except Exception as e:
        st.warning(f"nba_api Stats Fetch Note: {e}")
        return pd.DataFrame(), pd.DataFrame()

# --- 3. PACE, DVP & INJURY USAGE REALLOCATION ---
def apply_pace_and_matchups(df, pace_df):
    """Calculates Pace Factor for each matchup."""
    if pace_df.empty or "Game Info" not in df.columns:
        df["Pace_Factor"] = 1.0
        return df

    def extract_opp(row):
        game = str(row["Game Info"]).split(" ")[0]
        if "@" in game:
            away, home = game.split("@")
            return home if row["TeamAbbrev"] == away else away
        return None

    df["Opponent"] = df.apply(extract_opp, axis=1)
    league_pace = pace_df["PACE"].mean()
    pace_map = pace_df.set_index("TeamAbbrev")["PACE"].to_dict()

    def calc_pace_mult(row):
        t_pace = pace_map.get(row["TeamAbbrev"], league_pace)
        o_pace = pace_map.get(row["Opponent"], league_pace)
        expected_game_pace = (t_pace * o_pace) / league_pace
        return round(expected_game_pace / t_pace, 3)

    df["Pace_Factor"] = df.apply(calc_pace_mult, axis=1)
    return df

def reallocate_vacated_usage(df_pool, scratched_players):
    """Zeroes out scratched players and redistributes their minutes & usage to active teammates."""
    df = df_pool.copy()
    df["Is_Out"] = df["Name"].isin(scratched_players)
    
    # Baseline USG% approximation from salary & role
    df["USG%"] = np.clip(14.0 + (df["Salary"] - 3500) * (20.0 / 8500), 12.0, 36.0)
    df["FPPM"] = np.where(df["MIN"] > 0, df["Base_Proj"] / df["MIN"], df["Salary"] / 24000.0)

    adjusted = []
    for team, group in df.groupby("TeamAbbrev"):
        outs = group[group["Is_Out"]]
        actives = group[~group["Is_Out"]].copy()

        if outs.empty:
            actives["Projected_FPTS"] = (actives["Base_Proj"] * actives["Pace_Factor"]).round(2)
            adjusted.append(actives)
            continue

        vacated_min = outs["MIN"].sum()
        vacated_usg = outs["USG%"].sum()

        # Distribute minutes up to 36 max
        min_room = np.maximum(36.0 - actives["MIN"], 0)
        if min_room.sum() > 0:
            actives["Adj_MIN"] = np.minimum(
                actives["MIN"] + (vacated_min * (min_room / min_room.sum()) * 0.75), 38.0
            ).round(1)
        else:
            actives["Adj_MIN"] = actives["MIN"]

        # Distribute usage proportionally to higher-usage core
        usg_weights = actives["USG%"] / actives["USG%"].sum()
        actives["Adj_USG%"] = (actives["USG%"] + (vacated_usg * 0.70 * usg_weights)).round(2)
        usg_lift = (actives["Adj_USG%"] - actives["USG%"]) / actives["USG%"]
        
        # New FPPM boosted by usage bump + pace factor
        new_fppm = actives["FPPM"] * (1.0 + (usg_lift * 0.75))
        actives["Projected_FPTS"] = (actives["Adj_MIN"] * new_fppm * actives["Pace_Factor"]).round(2)
        adjusted.append(actives)

    result_df = pd.concat(adjusted, ignore_index=True)
    result_df["Value (x/1k)"] = (result_df["Projected_FPTS"] / (result_df["Salary"] / 1000)).round(2)
    return result_df

# --- 4. 5-LINEUP MULTI-ENTRY OPTIMIZER (PuLP) ---
def optimize_5_lineups(
    pool_df, 
    locks=None, 
    excludes=None, 
    cap=50000, 
    max_exposure=0.60, 
    target_stack=None, 
    stack_size=3
):
    """
    Solves for exactly 5 distinct GPP lineups iteratively while enforcing:
    - 8 DK Roster Slots (PG, SG, SF, PF, C, G, F, UTIL)
    - $50k Salary Cap
    - Min 2 unique real-world NBA games
    - Maximum player exposure ceiling across the 5 lineups
    - Game Stacking with Bring-Back
    """
    active = pool_df[~pool_df["Name"].isin(excludes or [])].copy().reset_index(drop=True)
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    def is_eligible(pos_str, slot):
        pos = [p.strip() for p in pos_str.split("/")]
        if slot == "UTIL":
            return True
        if slot == "G":
            return any(p in ["PG", "SG"] for p in pos)
        if slot == "F":
            return any(p in ["SF", "PF"] for p in pos)
        return slot in pos

    indices = list(active.index)
    num_lineups = 5
    max_appearances = max(1, int(round(num_lineups * max_exposure)))
    
    player_usage_count = {i: 0 for i in indices}
    lineup_history = []
    generated_lineups = []

    for l_idx in range(num_lineups):
        prob = pulp.LpProblem(f"DK_NBA_Opt_{l_idx}", pulp.LpMaximize)

        # Slot decision variables: x[i, s]
        x = {}
        for i in indices:
            for s in slots:
                if is_eligible(active.loc[i, "Position"], s):
                    x[i, s] = pulp.LpVariable(f"x_{l_idx}_{i}_{s}", cat="Binary")
                else:
                    x[i, s] = 0

        # Overall player selection indicator: y[i]
        y = {i: pulp.LpVariable(f"y_{l_idx}_{i}", cat="Binary") for i in indices}
        for i in indices:
            prob += pulp.lpSum(x[i, s] for s in slots) == y[i]

        # Objective: Maximize Projected Points
        prob += pulp.lpSum(active.loc[i, "Projected_FPTS"] * y[i] for i in indices)

        # Constraints
        for s in slots:
            prob += pulp.lpSum(x[i, s] for i in indices) == 1
        for i in indices:
            prob += pulp.lpSum(x[i, s] for s in slots) <= 1
        prob += pulp.lpSum(active.loc[i, "Salary"] * y[i] for i in indices) <= cap

        # Locks
        for lock in (locks or []):
            lock_idx = active[active["Name"] == lock].index
            if len(lock_idx) > 0:
                prob += y[lock_idx[0]] == 1

        # Max Exposure Cap across iterations
        for i in indices:
            if player_usage_count[i] >= max_appearances and active.loc[i, "Name"] not in (locks or []):
                prob += y[i] == 0

        # Ban exact duplicate lineups (overlap <= 6 players)
        for prev_y in lineup_history:
            prob += pulp.lpSum(y[i] for i in prev_y) <= 6

        # Min 2 Games constraint
        games = active["Game Info"].unique()
        g_vars = {g: pulp.LpVariable(f"g_{l_idx}_{g_i}", cat="Binary") for g_i, g in enumerate(games)}
        for g in games:
            g_idx = active[active["Game Info"] == g].index
            prob += pulp.lpSum(y[i] for i in g_idx) >= g_vars[g]
            prob += pulp.lpSum(y[i] for i in g_idx) <= 8 * g_vars[g]
        prob += pulp.lpSum(g_vars[g] for g in games) >= 2

        # Optional Game Stacking with Bring-Back
        if target_stack and target_stack != "None":
            stk_rows = active[active["Game Info"].str.contains(target_stack, case=False, na=False)]
            if len(stk_rows) >= stack_size:
                teams = stk_rows["TeamAbbrev"].unique()
                prob += pulp.lpSum(y[i] for i in stk_rows.index) >= stack_size
                if len(teams) == 2:
                    t1_idx = stk_rows[stk_rows["TeamAbbrev"] == teams[0]].index
                    t2_idx = stk_rows[stk_rows["TeamAbbrev"] == teams[1]].index
                    prob += pulp.lpSum(y[i] for i in t1_idx) >= 1
                    prob += pulp.lpSum(y[i] for i in t2_idx) >= 1

        # Solve
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if pulp.LpStatus[prob.status] != "Optimal":
            break

        # Extract selected roster
        selected_indices = [i for i in indices if pulp.value(y[i]) > 0.5]
        lineup_history.append(selected_indices)
        for i in selected_indices:
            player_usage_count[i] += 1

        roster_rows = []
        for s in slots:
            for i in indices:
                val = pulp.value(x[i, s])
                if val is not None and val > 0.5:
                    row = active.loc[i].to_dict()
                    row["Roster Slot"] = s
                    roster_rows.append(row)

        roster_df = pd.DataFrame(roster_rows)
        order = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4, "G": 5, "F": 6, "UTIL": 7}
        roster_df["Sort"] = roster_df["Roster Slot"].map(order)
        roster_df = roster_df.sort_values(by="Sort").drop(columns=["Sort"])
        roster_df["Lineup_Num"] = l_idx + 1
        generated_lineups.append(roster_df)

    return generated_lineups

# --- 5. STREAMLIT APPLICATION CONTROLS ---
st.sidebar.header("1. Live Slate Selection")
active_slates = fetch_dk_nba_contests()

if active_slates:
    selected_dg = st.sidebar.selectbox("Active Slates", options=list(active_slates.keys()), format_func=lambda x: active_slates[x])
else:
    selected_dg = st.sidebar.text_input("Manual DraftGroup ID", value="46589")

salary_cap = st.sidebar.number_input("Salary Cap ($)", value=50000, step=500)

with st.spinner("Fetching DraftKings Player Pool & Stats..."):
    slate_df = fetch_dk_players_by_draftgroup(selected_dg)
    nba_player_stats, team_pace_stats = fetch_nba_stats()

if slate_df.empty:
    st.error("No player data found for this slate.")
    st.stop()

# Merge stats & calculate baseline
if not nba_player_stats.empty:
    merged_df = pd.merge(slate_df, nba_player_stats, left_on="Name", right_on="PLAYER_NAME", how="left")
    merged_df["MIN"] = merged_df["MIN"].fillna(20.0).round(1)
    merged_df["Base_Proj"] = merged_df["Season_FPTS"].fillna(merged_df["Salary"] / 240.0).round(2)
else:
    merged_df = slate_df.copy()
    merged_df["MIN"] = 24.0
    merged_df["Base_Proj"] = (merged_df["Salary"] / 220.0).round(2)

# Apply pace factor
merged_df = apply_pace_and_matchups(merged_df, team_pace_stats)

# Sidebar: Injury Scratches
st.sidebar.header("2. Injury Status (Scratches)")
all_names = sorted(merged_df["Name"].unique().tolist())
outs = st.sidebar.multiselect("Mark OUT (Reallocates Usage/Minutes)", options=all_names)

# Recalculate projections dynamically based on scratches
pool_df = reallocate_vacated_usage(merged_df, outs)

# Sidebar: Tournament Controls
st.sidebar.header("3. GPP & Portfolio Controls")
locks = st.sidebar.multiselect("🔒 Lock Core Players", options=all_names)
fades = st.sidebar.multiselect("❌ Exclude Players", options=[p for p in all_names if p not in locks])

max_exp = st.sidebar.slider(
    "Max Player Exposure across 5 Lineups",
    min_value=0.20, max_value=1.0, value=0.60, step=0.20,
    help="0.60 means a player can appear in at most 3 of the 5 lineups."
)

slate_games = ["None"] + [g for g in pool_df["Game Info"].unique() if "@" in str(g)]
game_stack = st.sidebar.selectbox("🎯 Target Game Stack", options=slate_games)
stack_players = st.sidebar.slider("Min Players from Stacked Game", 2, 4, 3)

# --- 6. DISPLAY & EXECUTION TABS ---
tab_lineups, tab_portfolio, tab_pool = st.tabs(["⚡ 5 Optimized Lineups", "📊 Exposure Breakdown", "📋 Player Pool & Projections"])

with tab_lineups:
    if st.button("🚀 Generate 5 Tournament Lineups", type="primary"):
        with st.spinner("Solving multi-lineup linear integer program..."):
            five_lineups = optimize_5_lineups(
                pool_df=pool_df,
                locks=locks,
                excludes=fades,
                cap=salary_cap,
                max_exposure=max_exp,
                target_stack=game_stack,
                stack_size=stack_players
            )

        if len(five_lineups) < 5:
            st.warning(f"Solver generated {len(five_lineups)} valid lineups before running out of valid combinations. Try raising the maximum exposure limit or reducing player locks.")

        if five_lineups:
            # Combine for export
            export_rows = []
            for idx, l_df in enumerate(five_lineups):
                st.subheader(f"Lineup #{idx+1} — {l_df['Projected_FPTS'].sum().round(1)} Projected FPTS (${l_df['Salary'].sum():,} Salary)")
                st.dataframe(
                    l_df[["Roster Slot", "Name", "Position", "TeamAbbrev", "Salary", "Projected_FPTS", "Value (x/1k)", "Game Info"]],
                    use_container_width=True,
                    hide_index=True
                )
                
                # Format single row for DraftKings CSV upload format
                export_rows.append({
                    "PG": l_df[l_df["Roster Slot"] == "PG"]["Name"].values[0],
                    "SG": l_df[l_df["Roster Slot"] == "SG"]["Name"].values[0],
                    "SF": l_df[l_df["Roster Slot"] == "SF"]["Name"].values[0],
                    "PF": l_df[l_df["Roster Slot"] == "PF"]["Name"].values[0],
                    "C": l_df[l_df["Roster Slot"] == "C"]["Name"].values[0],
                    "G": l_df[l_df["Roster Slot"] == "G"]["Name"].values[0],
                    "F": l_df[l_df["Roster Slot"] == "F"]["Name"].values[0],
                    "UTIL": l_df[l_df["Roster Slot"] == "UTIL"]["Name"].values[0],
                })

            # CSV Download Button
            dk_export_df = pd.DataFrame(export_rows)
            csv_buf = io.StringIO()
            dk_export_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download 5-Lineup DraftKings CSV (Ready for Upload)",
                data=csv_buf.getvalue(),
                file_name="dk_5_lineup_entry.csv",
                mime="text/csv"
            )

            # Store in session state for portfolio tab
            st.session_state["five_lineups"] = five_lineups

with tab_portfolio:
    st.subheader("Player Portfolio & Leverage Exposures")
    if "five_lineups" in st.session_state:
        all_picks = pd.concat(st.session_state["five_lineups"])
        exp_df = all_picks["Name"].value_counts().reset_index()
        exp_df.columns = ["Name", "Count"]
        exp_df["Exposure %"] = ((exp_df["Count"] / 5) * 100).round(1)
        
        # Merge with Salary and Team
        meta = pool_df[["Name", "TeamAbbrev", "Position", "Salary", "Projected_FPTS"]].drop_duplicates()
        portfolio_summary = pd.merge(exp_df, meta, on="Name", how="left")
        
        st.dataframe(
            portfolio_summary[["Name", "TeamAbbrev", "Position", "Salary", "Count", "Exposure %", "Projected_FPTS"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Click 'Generate 5 Tournament Lineups' in the first tab to view portfolio distribution.")

with tab_pool:
    st.subheader("Active Player Pool (Post Pace & Injury Adjustments)")
    st.dataframe(
        pool_df[["Name", "Position", "TeamAbbrev", "Salary", "MIN", "Pace_Factor", "Projected_FPTS", "Value (x/1k)", "Game Info"]]
        .sort_values(by="Projected_FPTS", ascending=False),
        use_container_width=True,
        hide_index=True
    )
