import streamlit as st
import pandas as pd
import numpy as np
import requests
import pulp
import io
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

st.set_page_config(
    page_title="DraftKings NBA DFS 5-Lineup Engine",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 DraftKings NBA DFS 5-Lineup Multi-Entry Engine")
st.caption("Live Slate / CSV Ingestion • Pace & Usage Redistribution • Zero-Crash Linear Program")

# --- BROWSER HEADERS ---
DK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.draftkings.com/",
    "Origin": "https://www.draftkings.com"
}

# --- 1. SLATE DATA FETCHERS & FALLBACK ---
@st.cache_data(ttl=600)
def fetch_dk_nba_contests():
    """Scans DraftKings lobby for active NBA slates."""
    url = "https://www.draftkings.com/lobby/getcontests?sport=NBA"
    try:
        res = requests.get(url, headers=DK_HEADERS, timeout=8)
        if res.status_code != 200 or not res.text.strip():
            return {}
        data = res.json()
        contests = data.get("Contests", [])
        slates = {}
        for c in contests:
            d_id = c.get("dg")
            name = c.get("n", "NBA Slate")
            if d_id and d_id not in slates:
                slates[d_id] = f"Slate {d_id}: {name[:45]}"
        return slates
    except Exception:
        return {}

@st.cache_data(ttl=600)
def fetch_dk_players_by_draftgroup(draft_group_id):
    """Pulls full active player pool for a DraftGroup without crashing on 403/HTML."""
    if not draft_group_id:
        return pd.DataFrame()
    url = f"https://api.draftkings.com/draftgroups/v1/draftgroups/{draft_group_id}/draftables?format=json"
    try:
        res = requests.get(url, headers=DK_HEADERS, timeout=8)
        if res.status_code != 200 or not res.text.strip():
            return pd.DataFrame()
        data = res.json()
        draftables = data.get("draftables", [])
        if not draftables:
            return pd.DataFrame()

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
        return pd.DataFrame(records).drop_duplicates(subset=["Name", "Position"])
    except Exception:
        return pd.DataFrame()

def load_demo_slate():
    """Fallback interactive demo slate when cloud host IP is blocked by DraftKings firewall."""
    return pd.DataFrame([
        {"Name": "Luka Doncic", "Position": "PG", "Salary": 11600, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "MIN": 37.2, "Base_Proj": 58.4},
        {"Name": "Nikola Jokic", "Position": "C", "Salary": 11400, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "MIN": 35.8, "Base_Proj": 60.1},
        {"Name": "Shai Gilgeous-Alexander", "Position": "PG", "Salary": 10400, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "MIN": 34.6, "Base_Proj": 53.2},
        {"Name": "Anthony Davis", "Position": "PF/C", "Salary": 10100, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "MIN": 35.0, "Base_Proj": 51.5},
        {"Name": "LeBron James", "Position": "SF/PF", "Salary": 9400, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "MIN": 34.2, "Base_Proj": 46.8},
        {"Name": "Kyrie Irving", "Position": "PG/SG", "Salary": 8600, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "MIN": 34.0, "Base_Proj": 42.5},
        {"Name": "Chet Holmgren", "Position": "PF/C", "Salary": 7900, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "MIN": 32.1, "Base_Proj": 39.0},
        {"Name": "Jamal Murray", "Position": "PG", "Salary": 7600, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "MIN": 33.0, "Base_Proj": 37.0},
        {"Name": "Jalen Williams", "Position": "SG/SF", "Salary": 7200, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "MIN": 32.5, "Base_Proj": 36.2},
        {"Name": "Austin Reaves", "Position": "PG/SG", "Salary": 6200, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "MIN": 31.4, "Base_Proj": 31.0},
        {"Name": "Aaron Gordon", "Position": "PF", "Salary": 5800, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "MIN": 30.5, "Base_Proj": 29.5},
        {"Name": "P.J. Washington", "Position": "SF/PF", "Salary": 5200, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "MIN": 28.0, "Base_Proj": 26.0},
        {"Name": "Dereck Lively II", "Position": "C", "Salary": 4800, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "MIN": 24.5, "Base_Proj": 25.0},
        {"Name": "Christian Braun", "Position": "SG", "Salary": 4400, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "MIN": 26.0, "Base_Proj": 22.0},
        {"Name": "Cason Wallace", "Position": "PG/SG", "Salary": 3800, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "MIN": 20.0, "Base_Proj": 18.0},
    ])

# --- 2. NBA STATS & PACE ENGINE ---
@st.cache_data(ttl=3600)
def fetch_nba_stats():
    """Pulls player averages & team pace from official NBA stats."""
    try:
        dash = leaguedashplayerstats.LeagueDashPlayerStats(
            per_mode_detailed='PerGame',
            season='2025-26',
            season_type_all_star='Regular Season'
        )
        p_df = dash.get_data_frames()[0]

        p_df["Base_DK_FPTS"] = (
            p_df["PTS"] * 1.0 +
            p_df["FG3M"] * 0.5 +
            p_df["REB"] * 1.25 +
            p_df["AST"] * 1.5 +
            p_df["STL"] * 2.0 +
            p_df["BLK"] * 2.0 -
            p_df["TOV"] * 0.5
        )
        p_df["DD_Bonus"] = (
            ((p_df["PTS"] >= 10).astype(int) + 
             (p_df["REB"] >= 10).astype(int) + 
             (p_df["AST"] >= 10).astype(int)) >= 2
        ).astype(int) * 1.5

        p_df["Season_FPTS"] = (p_df["Base_DK_FPTS"] + p_df["DD_Bonus"]).round(2)
        player_clean = p_df[["PLAYER_NAME", "MIN", "Season_FPTS"]].copy()

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
        team_clean = t_df[["TeamAbbrev", "PACE"]].dropna()

        return player_clean, team_clean
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def apply_pace_and_matchups(df, pace_df):
    """Calculates Pace Multiplier based on expected possessions."""
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

        min_room = np.maximum(36.0 - actives["MIN"], 0)
        if min_room.sum() > 0:
            actives["Adj_MIN"] = np.minimum(
                actives["MIN"] + (vacated_min * (min_room / min_room.sum()) * 0.75), 38.0
            ).round(1)
        else:
            actives["Adj_MIN"] = actives["MIN"]

        usg_weights = actives["USG%"] / actives["USG%"].sum()
        actives["Adj_USG%"] = (actives["USG%"] + (vacated_usg * 0.70 * usg_weights)).round(2)
        usg_lift = (actives["Adj_USG%"] - actives["USG%"]) / actives["USG%"]
        
        new_fppm = actives["FPPM"] * (1.0 + (usg_lift * 0.75))
        actives["Projected_FPTS"] = (actives["Adj_MIN"] * new_fppm * actives["Pace_Factor"]).round(2)
        adjusted.append(actives)

    result_df = pd.concat(adjusted, ignore_index=True)
    result_df["Value (x/1k)"] = (result_df["Projected_FPTS"] / (result_df["Salary"] / 1000)).round(2)
    return result_df

# --- 3. PYTHON 3.14 COMPATIBLE 5-LINEUP OPTIMIZER ---
def optimize_5_lineups(pool_df, locks=None, excludes=None, cap=50000, max_exposure=0.60, target_stack=None, stack_size=3):
    active = pool_df[~pool_df["Name"].isin(excludes or [])].copy().reset_index(drop=True)
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]

    def is_eligible(pos_str, slot):
        pos = [p.strip() for p in str(pos_str).split("/")]
        if slot == "UTIL":
            return True
        if slot == "G":
            return any(p in ["PG", "SG"] for p in pos)
        if slot == "F":
            return any(p in ["SF", "PF"] for p in pos)
        return slot in pos

    n_players = len(active)
    indices = list(range(n_players))
    num_lineups = 5
    max_appearances = max(1, int(round(num_lineups * max_exposure)))
    
    player_usage_count = {i: 0 for i in indices}
    lineup_history = []
    generated_lineups = []

    for l_idx in range(num_lineups):
        prob = pulp.LpProblem(f"DK_Lineup_{l_idx}", pulp.LpMaximize)

        # Python 3.14 safe integer variable declaration (0 to 1 bounds avoid LpVariable constructor bugs)
        x = {}
        for i in indices:
            x[i] = {}
            for s in slots:
                if is_eligible(active.loc[i, "Position"], s):
                    var_name = f"x_{l_idx}_{i}_{s}"
                    x[i][s] = pulp.LpVariable(name=var_name, lowBound=0, upBound=1, cat="Integer")
                else:
                    x[i][s] = 0

        y = {}
        for i in indices:
            y[i] = pulp.LpVariable(name=f"y_{l_idx}_{i}", lowBound=0, upBound=1, cat="Integer")

        # Link player indicator to slot usage
        for i in indices:
            prob += pulp.lpSum(x[i][s] for s in slots) == y[i]

        # Maximize total projected points
        prob += pulp.lpSum(float(active.loc[i, "Projected_FPTS"]) * y[i] for i in indices)

        # 1 player per roster slot
        for s in slots:
            prob += pulp.lpSum(x[i][s] for i in indices) == 1

        # Max 1 selection per player
        for i in indices:
            prob += pulp.lpSum(x[i][s] for s in slots) <= 1

        # Salary cap
        prob += pulp.lpSum(int(active.loc[i, "Salary"]) * y[i] for i in indices) <= cap

        # Lock core players
        for lock in (locks or []):
            lock_idx = active[active["Name"] == lock].index
            if len(lock_idx) > 0:
                prob += y[lock_idx[0]] == 1

        # Max exposure cap across lineups
        for i in indices:
            if player_usage_count[i] >= max_appearances and active.loc[i, "Name"] not in (locks or []):
                prob += y[i] == 0

        # Overlap limit: Lineups must differ by at least 2 players
        for prev_y in lineup_history:
            prob += pulp.lpSum(y[i] for i in prev_y) <= 6

        # Min 2 Games constraint
        games = [g for g in active["Game Info"].unique() if pd.notna(g)]
        g_vars = {}
        for g_idx, g in enumerate(games):
            g_vars[g_idx] = pulp.LpVariable(name=f"g_{l_idx}_{g_idx}", lowBound=0, upBound=1, cat="Integer")
            p_in_game = active[active["Game Info"] == g].index
            prob += pulp.lpSum(y[i] for i in p_in_game) >= g_vars[g_idx]
            prob += pulp.lpSum(y[i] for i in p_in_game) <= 8 * g_vars[g_idx]

        prob += pulp.lpSum(g_vars[g_idx] for g_idx in range(len(games))) >= 2

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

        selected_indices = [i for i in indices if pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5]
        lineup_history.append(selected_indices)
        for i in selected_indices:
            player_usage_count[i] += 1

        roster_rows = []
        for s in slots:
            for i in indices:
                val = pulp.value(x[i][s])
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

# --- 4. SIDEBAR & USER CONTROLS ---
st.sidebar.header("1. Slate Data Ingestion")
data_mode = st.sidebar.radio("Data Ingestion Mode", ["Live DraftKings API", "Upload DKSalaries.csv"])

slate_df = pd.DataFrame()

if data_mode == "Live DraftKings API":
    active_slates = fetch_dk_nba_contests()
    if active_slates:
        selected_dg = st.sidebar.selectbox("Active Slates", options=list(active_slates.keys()), format_func=lambda x: active_slates[x])
        with st.spinner("Fetching slate from DraftKings API..."):
            slate_df = fetch_dk_players_by_draftgroup(selected_dg)
    else:
        st.sidebar.warning("⚠️ Live API restricted by host IP. Upload DKSalaries.csv or use demo slate.")

elif data_mode == "Upload DKSalaries.csv":
    uploaded_file = st.sidebar.file_uploader("Upload DKSalaries.csv", type=["csv"])
    if uploaded_file is not None:
        slate_df = pd.read_csv(uploaded_file)
        if "Game Info" not in slate_df.columns:
            slate_df["Game Info"] = "NBA Game"

if slate_df.empty:
    st.info("💡 Live DraftKings API is blocked or no slate was found. Loaded interactive demo slate. Upload `DKSalaries.csv` in the sidebar to run today's live contests.")
    slate_df = load_demo_slate()

salary_cap = st.sidebar.number_input("Salary Cap ($)", value=50000, step=500)

with st.spinner("Calculating projections & pace metrics..."):
    nba_player_stats, team_pace_stats = fetch_nba_stats()

# Merge stats & calculate baseline
if not nba_player_stats.empty:
    merged_df = pd.merge(slate_df, nba_player_stats, left_on="Name", right_on="PLAYER_NAME", how="left")
    merged_df["MIN"] = merged_df["MIN"].fillna(20.0).round(1)
    if "Base_Proj" not in merged_df.columns:
        merged_df["Base_Proj"] = merged_df["Season_FPTS"].fillna(merged_df["Salary"] / 240.0).round(2)
else:
    merged_df = slate_df.copy()
    if "MIN" not in merged_df.columns:
        merged_df["MIN"] = 24.0
    if "Base_Proj" not in merged_df.columns:
        merged_df["Base_Proj"] = (merged_df["Salary"] / 220.0).round(2)

merged_df = apply_pace_and_matchups(merged_df, team_pace_stats)

# Sidebar: Injury Scratches
st.sidebar.header("2. Injury Status (Scratches)")
all_names = sorted(merged_df["Name"].unique().tolist())
outs = st.sidebar.multiselect("Mark OUT (Reallocates Usage/Minutes)", options=all_names)

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

# --- 5. TABS INTERFACE ---
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
            export_rows = []
            for idx, l_df in enumerate(five_lineups):
                st.subheader(f"Lineup #{idx+1} — {l_df['Projected_FPTS'].sum().round(1)} Projected FPTS (${l_df['Salary'].sum():,} Salary)")
                st.dataframe(
                    l_df[["Roster Slot", "Name", "Position", "TeamAbbrev", "Salary", "Projected_FPTS", "Value (x/1k)", "Game Info"]],
                    use_container_width=True,
                    hide_index=True
                )
                
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

            dk_export_df = pd.DataFrame(export_rows)
            csv_buf = io.StringIO()
            dk_export_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download 5-Lineup DraftKings CSV (Ready for Upload)",
                data=csv_buf.getvalue(),
                file_name="dk_5_lineup_entry.csv",
                mime="text/csv"
            )

            st.session_state["five_lineups"] = five_lineups

with tab_portfolio:
    st.subheader("Player Portfolio & Leverage Exposures")
    if "five_lineups" in st.session_state:
        all_picks = pd.concat(st.session_state["five_lineups"])
        exp_df = all_picks["Name"].value_counts().reset_index()
        exp_df.columns = ["Name", "Count"]
        exp_df["Exposure %"] = ((exp_df["Count"] / 5) * 100).round(1)
        
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
