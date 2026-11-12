import streamlit as st
import pandas as pd
import numpy as np
import requests
import io
from scipy.optimize import milp, LinearConstraint, Bounds
from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

st.set_page_config(
    page_title="DraftKings NBA DFS 5-Lineup Engine",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 DraftKings NBA DFS 5-Lineup Multi-Entry Engine")
st.caption("Vegas Implied Totals & Blowout Protection • Normalized Game Caps • 75th/90th Ceiling • SciPy MILP")

DK_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.draftkings.com/",
    "Origin": "https://www.draftkings.com"
}

TEAM_NAME_TO_ABBREV = {
    'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
    'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
    'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
    'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
    'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
    'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI',
    'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC',
    'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA',
    'Washington Wizards': 'WAS'
}

def normalize_team_abbrev(abbrev):
    """Standardizes DK team codes (e.g., GS -> GSW, NO -> NOP, SA -> SAS, NY -> NYK, PHO -> PHX)."""
    mapping = {
        "GS": "GSW", "NO": "NOP", "NY": "NYK", "SA": "SAS",
        "PHO": "PHX", "UTAH": "UTA", "WSH": "WAS", "BK": "BKN"
    }
    clean = str(abbrev).strip().upper()
    return mapping.get(clean, clean)

def clean_game_string(game_str):
    """Strips DK timestamps so 'LAL@ATL 11/12/2026 07:30PM ET' becomes 'LAL@ATL'."""
    raw = str(game_str).strip().split(" ")[0].upper()
    if "@" in raw:
        parts = raw.split("@")
        if len(parts) == 2:
            away = normalize_team_abbrev(parts[0])
            home = normalize_team_abbrev(parts[1])
            return f"{away}@{home}"
    return raw

# --- 1. SLATE DATA FETCHERS & FALLBACK ---
@st.cache_data(ttl=600)
def fetch_dk_nba_contests():
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
            game_desc = comp.get("name", "")
            team = normalize_team_abbrev(p.get("teamAbbreviation", ""))
            if not game_desc:
                game_desc = f"{team}_GAME"

            records.append({
                "PlayerId": p.get("playerId"),
                "Name": p.get("displayName"),
                "Position": p.get("position", ""),
                "Salary": int(p.get("salary", 0)),
                "TeamAbbrev": team,
                "Game Info": game_desc,
                "Instructions": p.get("status", "")
            })
        return pd.DataFrame(records).drop_duplicates(subset=["Name", "Position"])
    except Exception:
        return pd.DataFrame()

def load_demo_slate():
    return pd.DataFrame([
        {"Name": "Luka Doncic", "Position": "PG", "Salary": 11600, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 37.2, "Base_Proj": 58.4},
        {"Name": "Nikola Jokic", "Position": "C", "Salary": 11400, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 35.8, "Base_Proj": 60.1},
        {"Name": "Shai Gilgeous-Alexander", "Position": "PG", "Salary": 10400, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 34.6, "Base_Proj": 53.2},
        {"Name": "Anthony Davis", "Position": "PF/C", "Salary": 10100, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 35.0, "Base_Proj": 51.5},
        {"Name": "LeBron James", "Position": "SF/PF", "Salary": 9400, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 34.2, "Base_Proj": 46.8},
        {"Name": "Kyrie Irving", "Position": "PG/SG", "Salary": 8600, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 34.0, "Base_Proj": 42.5},
        {"Name": "Chet Holmgren", "Position": "PF/C", "Salary": 7900, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 32.1, "Base_Proj": 39.0},
        {"Name": "Jamal Murray", "Position": "PG", "Salary": 7600, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 33.0, "Base_Proj": 37.0},
        {"Name": "Jalen Williams", "Position": "SG/SF", "Salary": 7200, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 32.5, "Base_Proj": 36.2},
        {"Name": "Austin Reaves", "Position": "PG/SG", "Salary": 6200, "TeamAbbrev": "LAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 31.4, "Base_Proj": 31.0},
        {"Name": "Aaron Gordon", "Position": "PF", "Salary": 5800, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 30.5, "Base_Proj": 29.5},
        {"Name": "P.J. Washington", "Position": "SF/PF", "Salary": 5200, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 28.0, "Base_Proj": 26.0},
        {"Name": "Dereck Lively II", "Position": "C", "Salary": 4800, "TeamAbbrev": "DAL", "Game Info": "DAL@LAL", "Instructions": "", "MIN": 24.5, "Base_Proj": 25.0},
        {"Name": "Christian Braun", "Position": "SG", "Salary": 4400, "TeamAbbrev": "DEN", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 26.0, "Base_Proj": 22.0},
        {"Name": "Cason Wallace", "Position": "PG/SG", "Salary": 3800, "TeamAbbrev": "OKC", "Game Info": "OKC@DEN", "Instructions": "", "MIN": 20.0, "Base_Proj": 18.0},
    ])

# --- 2. LIVE VEGAS ODDS FETCHER (THE ODDS API) ---
@st.cache_data(ttl=900)
def fetch_vegas_odds(api_key):
    """Fetches live NBA spreads and game totals from The Odds API."""
    if not api_key or not api_key.strip():
        return {}
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": api_key.strip(),
        "regions": "us",
        "markets": "spreads,totals",
        "oddsFormat": "american"
    }
    try:
        res = requests.get(url, params=params, timeout=10)
        if res.status_code != 200:
            return {}
        data = res.json()
        odds_map = {}
        for event in data:
            away_abbr = TEAM_NAME_TO_ABBREV.get(event.get("away_team"))
            home_abbr = TEAM_NAME_TO_ABBREV.get(event.get("home_team"))
            if not away_abbr or not home_abbr:
                continue
            game_key = f"{away_abbr}@{home_abbr}"
            total_val = 228.0
            home_spread = 0.0
            bookmakers = event.get("bookmakers", [])
            if bookmakers:
                for mkt in bookmakers[0].get("markets", []):
                    if mkt["key"] == "totals" and mkt.get("outcomes"):
                        total_val = float(mkt["outcomes"][0].get("point", 228.0))
                    elif mkt["key"] == "spreads":
                        for out in mkt.get("outcomes", []):
                            if out.get("name") == event.get("home_team"):
                                home_spread = float(out.get("point", 0.0))
            odds_map[game_key] = {"Total": total_val, "Home_Spread": home_spread}
        return odds_map
    except Exception:
        return {}

# --- 3. OPEN-SOURCE CONSENSUS PROJECTIONS ---
@st.cache_data(ttl=900)
def fetch_open_source_projections():
    url = "https://www.dailyfantasyfuel.com/nba/projections/draftkings"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code != 200:
            return pd.DataFrame()
        tables = pd.read_html(io.StringIO(res.text))
        if not tables:
            return pd.DataFrame()

        proj_df = tables[0]
        for col in proj_df.columns:
            c_low = str(col).lower()
            if "player" in c_low or "name" in c_low:
                proj_df = proj_df.rename(columns={col: "Name"})
            elif "min" in c_low:
                proj_df = proj_df.rename(columns={col: "Open_MIN"})
            elif "proj" in c_low or "fpts" in c_low or "pts" in c_low:
                proj_df = proj_df.rename(columns={col: "Open_Proj_FPTS"})

        if "Name" not in proj_df.columns or "Open_Proj_FPTS" not in proj_df.columns:
            return pd.DataFrame()

        proj_df["Name"] = proj_df["Name"].astype(str).str.replace(r"[^a-zA-Z\s\.\']", "", regex=True).str.strip()
        proj_df["Open_MIN"] = pd.to_numeric(proj_df.get("Open_MIN", 0.0), errors="coerce").fillna(0.0)
        proj_df["Open_Proj_FPTS"] = pd.to_numeric(proj_df.get("Open_Proj_FPTS", 0.0), errors="coerce").fillna(0.0)

        return proj_df[["Name", "Open_MIN", "Open_Proj_FPTS"]].drop_duplicates(subset=["Name"])
    except Exception:
        return pd.DataFrame()

# --- 4. NBA STATS & VEGAS/PACE ADJUSTMENTS ---
@st.cache_data(ttl=3600)
def fetch_nba_stats():
    for season_str in ["2026-27", "2025-26"]:
        try:
            dash = leaguedashplayerstats.LeagueDashPlayerStats(
                per_mode_detailed='PerGame',
                season=season_str,
                season_type_all_star='Regular Season'
            )
            p_df = dash.get_data_frames()[0]
            if p_df.empty:
                continue

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
                season=season_str,
                season_type_all_star='Regular Season'
            )
            t_df = t_dash.get_data_frames()[0]
            t_df["TeamAbbrev"] = t_df["TEAM_NAME"].map(TEAM_NAME_TO_ABBREV)
            team_clean = t_df[["TeamAbbrev", "PACE"]].dropna()

            return player_clean, team_clean
        except Exception:
            continue
    return pd.DataFrame(), pd.DataFrame()

def apply_vegas_and_pace(df, pace_df, vegas_df, blowout_spread=11.5, blowout_discount=0.09):
    """Applies team pace, Vegas Implied Team Totals (ITT), and stud blowout haircuts."""
    df = df.copy()

    def extract_opp(row):
        game = str(row["Clean_Game"])
        if "@" in game:
            away, home = game.split("@")
            return home if row["TeamAbbrev"] == away else away
        return None

    df["Opponent"] = df.apply(extract_opp, axis=1)

    # 1. Pace Multiplier
    if not pace_df.empty:
        league_pace = pace_df["PACE"].mean()
        pace_map = pace_df.set_index("TeamAbbrev")["PACE"].to_dict()

        def calc_pace_mult(row):
            t_pace = pace_map.get(row["TeamAbbrev"], league_pace)
            o_pace = pace_map.get(row["Opponent"], league_pace)
            expected_game_pace = (t_pace * o_pace) / league_pace
            return round(expected_game_pace / t_pace, 3)

        df["Pace_Factor"] = df.apply(calc_pace_mult, axis=1)
    else:
        df["Pace_Factor"] = 1.0

    # 2. Vegas Implied Team Totals & Blowout Risk
    vegas_lookup = {}
    if not vegas_df.empty:
        for _, vrow in vegas_df.iterrows():
            g = vrow["Game"]
            if "@" in g:
                away, home = g.split("@")
                tot = float(vrow["Total"])
                h_sprd = float(vrow["Home_Spread"])
                home_itt = (tot / 2.0) - (h_sprd / 2.0)
                away_itt = (tot / 2.0) + (h_sprd / 2.0)
                vegas_lookup[home] = {"ITT": round(home_itt, 1), "Abs_Spread": abs(h_sprd)}
                vegas_lookup[away] = {"ITT": round(away_itt, 1), "Abs_Spread": abs(h_sprd)}

    def calc_vegas_metrics(row):
        info = vegas_lookup.get(row["TeamAbbrev"], {"ITT": 114.0, "Abs_Spread": 0.0})
        itt = info["ITT"]
        abs_spread = info["Abs_Spread"]
        # Dampened 50% ITT scaling around 114.0 league baseline
        itt_mult = 1.0 + ((itt - 114.0) / 114.0) * 0.50
        # Blowout penalty for $9,000+ stars when spread >= threshold
        b_mult = (1.0 - blowout_discount) if (abs_spread >= blowout_spread and row["Salary"] >= 9000) else 1.0
        return pd.Series([itt, abs_spread, round(itt_mult * b_mult, 3)])

    df[["Vegas_ITT", "Game_Spread", "Vegas_Factor"]] = df.apply(calc_vegas_metrics, axis=1)
    df["Env_Mult"] = (df["Pace_Factor"] * df["Vegas_Factor"]).round(3)
    return df

def reallocate_vacated_usage(df_pool, scratched_players):
    df = df_pool.copy()
    df["Is_Out"] = df["Name"].isin(scratched_players)
    df["USG%"] = np.clip(14.0 + (df["Salary"] - 3500) * (20.0 / 8500), 12.0, 36.0)
    df["FPPM"] = np.where(df["MIN"] > 0, df["Base_Proj"] / df["MIN"], df["Salary"] / 24000.0)

    adjusted = []
    for team, group in df.groupby("TeamAbbrev"):
        outs = group[group["Is_Out"]]
        actives = group[~group["Is_Out"]].copy()

        if outs.empty:
            actives["Projected_FPTS"] = (actives["Base_Proj"] * actives["Env_Mult"]).round(2)
            actives["Adj_MIN"] = actives["MIN"]
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
        actives["Projected_FPTS"] = (actives["Adj_MIN"] * new_fppm * actives["Env_Mult"]).round(2)
        adjusted.append(actives)

    if adjusted:
        result_df = pd.concat(adjusted, ignore_index=True)
    else:
        result_df = df.copy()
        result_df["Adj_MIN"] = result_df["MIN"]
        result_df["Projected_FPTS"] = result_df["Base_Proj"]

    return result_df

def calculate_percentile_ceilings(df):
    df = df.copy()
    min_factor = np.clip((36.0 - df["Adj_MIN"]) / 36.0, 0.0, 0.5)
    cv = 0.20 + (min_factor * 0.15)

    df["StdDev"] = (df["Projected_FPTS"] * cv).round(2)
    df["Ceiling_75"] = (df["Projected_FPTS"] + 0.674 * df["StdDev"]).round(2)
    df["Ceiling_90"] = (df["Projected_FPTS"] + 1.282 * df["StdDev"]).round(2)
    df["Value (x/1k)"] = (df["Projected_FPTS"] / (df["Salary"] / 1000)).round(2)
    return df

# --- 5. SCIPY MILP SOLVER ---
def optimize_5_lineups_scipy(
    pool_df,
    locks=None,
    excludes=None,
    cap=50000,
    max_exposure=0.60,
    target_stack=None,
    stack_size=3,
    max_game_players=4,
    objective_metric="Ceiling_90",
    min_studs=1,
    min_stud_salary=9500
):
    active = pool_df[~pool_df["Name"].isin(excludes or [])].copy().reset_index(drop=True)
    slots = ["PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"]
    n_slots = len(slots)
    n_players = len(active)

    if n_players < 8:
        return []

    def is_eligible(pos_str, slot):
        pos = [p.strip() for p in str(pos_str).split("/")]
        if slot == "UTIL":
            return True
        if slot == "G":
            return any(p in ["PG", "SG"] for p in pos)
        if slot == "F":
            return any(p in ["SF", "PF"] for p in pos)
        return slot in pos

    E = np.zeros((n_players, n_slots), dtype=int)
    for i in range(n_players):
        for s_idx, s in enumerate(slots):
            if is_eligible(active.loc[i, "Position"], s):
                E[i, s_idx] = 1

    games = [g for g in active["Clean_Game"].unique() if "@" in str(g)]
    n_games = len(games)

    total_vars = (n_players * n_slots) + n_players + n_games

    def var_x(i, s): return i * n_slots + s
    def var_y(i): return (n_players * n_slots) + i
    def var_g(g_idx): return (n_players * n_slots) + n_players + g_idx

    c = np.zeros(total_vars)
    for i in range(n_players):
        c[var_y(i)] = -float(active.loc[i, objective_metric])

    ub = np.ones(total_vars)
    lb = np.zeros(total_vars)

    for i in range(n_players):
        for s in range(n_slots):
            if E[i, s] == 0:
                ub[var_x(i, s)] = 0

    for lock in (locks or []):
        lock_idx = active[active["Name"] == lock].index
        if len(lock_idx) > 0:
            lb[var_y(lock_idx[0])] = 1

    rows, lower, upper = [], [], []

    # 1. Exactly 1 player per slot
    for s in range(n_slots):
        row = np.zeros(total_vars)
        for i in range(n_players):
            row[var_x(i, s)] = 1
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)

    # 2. Linking slot to player
    for i in range(n_players):
        row = np.zeros(total_vars)
        for s in range(n_slots):
            row[var_x(i, s)] = 1
        row[var_y(i)] = -1
        rows.append(row)
        lower.append(0.0)
        upper.append(0.0)

    # 3. Salary cap
    row = np.zeros(total_vars)
    for i in range(n_players):
        row[var_y(i)] = int(active.loc[i, "Salary"])
    rows.append(row)
    lower.append(0.0)
    upper.append(float(cap))

    # 4. Enforce Anchor Studs ($9,500+)
    stud_indices = active[active["Salary"] >= min_stud_salary].index
    if len(stud_indices) >= min_studs and min_studs > 0:
        row = np.zeros(total_vars)
        for i in stud_indices:
            row[var_y(i)] = 1
        rows.append(row)
        lower.append(float(min_studs))
        upper.append(np.inf)

    # 5. Unique game linking + Strict Anti-Cannibalization Game Cap
    if n_games >= 2:
        for g_idx, g in enumerate(games):
            p_indices = active[active["Clean_Game"] == g].index

            row1 = np.zeros(total_vars)
            for i in p_indices:
                row1[var_y(i)] = 1
            row1[var_g(g_idx)] = -1
            rows.append(row1)
            lower.append(0.0)
            upper.append(np.inf)

            row2 = np.zeros(total_vars)
            for i in p_indices:
                row2[var_y(i)] = 1
            row2[var_g(g_idx)] = -8
            rows.append(row2)
            lower.append(-np.inf)
            upper.append(0.0)

            # Hard Ceiling: Never allow more than max_game_players from one game
            row_cap = np.zeros(total_vars)
            for i in p_indices:
                row_cap[var_y(i)] = 1
            rows.append(row_cap)
            lower.append(0.0)
            upper.append(float(max_game_players))

        row = np.zeros(total_vars)
        for g_idx in range(n_games):
            row[var_g(g_idx)] = 1
        rows.append(row)
        lower.append(2.0)
        upper.append(np.inf)

    # 6. Target game stack with bring-back
    if target_stack and target_stack != "None" and "@" in str(target_stack):
        effective_stack = min(stack_size, max_game_players)
        stk_rows = active[active["Clean_Game"] == target_stack]
        if len(stk_rows) >= effective_stack:
            row = np.zeros(total_vars)
            for i in stk_rows.index:
                row[var_y(i)] = 1
            rows.append(row)
            lower.append(float(effective_stack))
            upper.append(np.inf)

            teams = stk_rows["TeamAbbrev"].unique()
            if len(teams) == 2:
                for t in teams:
                    t_idx = stk_rows[stk_rows["TeamAbbrev"] == t].index
                    row_t = np.zeros(total_vars)
                    for i in t_idx:
                        row_t[var_y(i)] = 1
                    rows.append(row_t)
                    lower.append(1.0)
                    upper.append(np.inf)

    num_lineups = 5
    max_appearances = max(1, int(round(num_lineups * max_exposure)))
    player_usage_count = np.zeros(n_players, dtype=int)
    history_rows = []
    generated_lineups = []

    for l_idx in range(num_lineups):
        iter_rows = list(rows)
        iter_lower = list(lower)
        iter_upper = list(upper)
        current_ub = np.array(ub)

        for i in range(n_players):
            if player_usage_count[i] >= max_appearances and active.loc[i, "Name"] not in (locks or []):
                current_ub[var_y(i)] = 0

        for prev_y in history_rows:
            row = np.zeros(total_vars)
            for i in prev_y:
                row[var_y(i)] = 1
            iter_rows.append(row)
            iter_lower.append(0.0)
            iter_upper.append(6.0)

        A = np.array(iter_rows)
        constraints = LinearConstraint(A, iter_lower, iter_upper)
        bounds = Bounds(lb, current_ub)
        integrality = np.ones(total_vars)

        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

        if not res.success:
            break

        sol = res.x
        selected_i = [i for i in range(n_players) if sol[var_y(i)] > 0.5]
        history_rows.append(selected_i)
        for i in selected_i:
            player_usage_count[i] += 1

        roster_rows = []
        for s_idx, s in enumerate(slots):
            for i in range(n_players):
                if sol[var_x(i, s_idx)] > 0.5:
                    r = active.loc[i].to_dict()
                    r["Roster Slot"] = s
                    roster_rows.append(r)

        roster_df = pd.DataFrame(roster_rows)
        order = {"PG": 0, "SG": 1, "SF": 2, "PF": 3, "C": 4, "G": 5, "F": 6, "UTIL": 7}
        roster_df["Sort"] = roster_df["Roster Slot"].map(order)
        roster_df = roster_df.sort_values(by="Sort").drop(columns=["Sort"])
        roster_df["Lineup_Num"] = l_idx + 1
        generated_lineups.append(roster_df)

    return generated_lineups

# --- 6. SIDEBAR & INTERFACE ---
st.sidebar.header("1. Slate Data Ingestion")
data_mode = st.sidebar.radio("Data Ingestion Mode", ["Upload DKSalaries.csv", "Live DraftKings API"])

slate_df = pd.DataFrame()

if data_mode == "Upload DKSalaries.csv":
    uploaded_file = st.sidebar.file_uploader("Upload DKSalaries.csv", type=["csv"])
    if uploaded_file is not None:
        slate_df = pd.read_csv(uploaded_file)
        if "Game Info" not in slate_df.columns:
            slate_df["Game Info"] = "NBA Game"

elif data_mode == "Live DraftKings API":
    active_slates = fetch_dk_nba_contests()
    if active_slates:
        selected_dg = st.sidebar.selectbox("Active Slates", options=list(active_slates.keys()), format_func=lambda x: active_slates[x])
        with st.spinner("Fetching slate from DraftKings API..."):
            slate_df = fetch_dk_players_by_draftgroup(selected_dg)
    else:
        st.sidebar.warning("⚠️ Live API restricted by host IP. Upload DKSalaries.csv or use demo slate.")

if slate_df.empty:
    st.info("💡 Upload your slate `DKSalaries.csv` in the sidebar to run live contests.")
    slate_df = load_demo_slate()

# Normalize team abbreviations and clean Game Info strings immediately
slate_df["TeamAbbrev"] = slate_df["TeamAbbrev"].apply(normalize_team_abbrev)
slate_df["Clean_Game"] = slate_df["Game Info"].apply(clean_game_string)

salary_cap = st.sidebar.number_input("Salary Cap ($)", value=50000, step=500)

# Sidebar: Vegas Odds & Blowout Protection
st.sidebar.header("2. Vegas Odds & Blowout Filter")
odds_api_key = st.sidebar.text_input("The Odds API Key (Optional)", type="password", help="Paste a free key from the-odds-api.com to auto-fill live spreads & totals.")
live_odds = fetch_vegas_odds(odds_api_key)

slate_matchups = sorted([g for g in slate_df["Clean_Game"].unique() if "@" in g])
if not slate_matchups:
    slate_matchups = ["DAL@LAL", "OKC@DEN"]

default_vegas_rows = []
for g in slate_matchups:
    g_odds = live_odds.get(g, {"Total": 228.0, "Home_Spread": 0.0})
    default_vegas_rows.append({
        "Game": g,
        "Total": float(g_odds["Total"]),
        "Home_Spread": float(g_odds["Home_Spread"])
    })

st.sidebar.caption("Edit Totals & Home Spreads (- = Home Favored, + = Away Favored):")
vegas_edited_df = st.sidebar.data_editor(
    pd.DataFrame(default_vegas_rows),
    hide_index=True,
    use_container_width=True,
    column_config={
        "Game": st.column_config.TextColumn("Game", disabled=True),
        "Total": st.column_config.NumberColumn("O/U", min_value=180.0, max_value=270.0, step=0.5),
        "Home_Spread": st.column_config.NumberColumn("Home Sprd", min_value=-30.0, max_value=30.0, step=0.5)
    }
)

blowout_threshold = st.sidebar.slider("Blowout Spread Threshold", min_value=8.0, max_value=18.0, value=11.5, step=0.5)

with st.spinner("Calculating consensus projections, Vegas ITT & pace metrics..."):
    nba_player_stats, team_pace_stats = fetch_nba_stats()
    open_proj_df = fetch_open_source_projections()

if not nba_player_stats.empty:
    merged_df = pd.merge(slate_df, nba_player_stats, left_on="Name", right_on="PLAYER_NAME", how="left")
else:
    merged_df = slate_df.copy()
    merged_df["MIN"] = 0.0
    merged_df["Season_FPTS"] = 0.0

if not open_proj_df.empty:
    merged_df = pd.merge(merged_df, open_proj_df, on="Name", how="left")
    merged_df["MIN"] = merged_df["Open_MIN"].fillna(merged_df.get("MIN", 0.0)).fillna(0.0).round(1)

    has_open = merged_df["Open_Proj_FPTS"].notna() & (merged_df["Open_Proj_FPTS"] > 0)
    merged_df["Base_Proj"] = np.where(
        has_open,
        (merged_df["Open_Proj_FPTS"] * 0.65) + (merged_df["Season_FPTS"].replace(0, np.nan).fillna(merged_df["Open_Proj_FPTS"]) * 0.35),
        merged_df["Season_FPTS"].replace(0, np.nan).fillna(merged_df["Salary"] / 240.0)
    ).round(2)
else:
    merged_df["MIN"] = merged_df.get("MIN", 0.0).fillna(0.0).round(1)
    merged_df["Base_Proj"] = merged_df.get("Season_FPTS", 0.0).replace(0, np.nan).fillna(merged_df["Salary"] / 240.0).round(2)

# Dynamic baseline floor for rotation starters
imputed_min = np.clip((merged_df["Salary"] - 3000) / 250.0, 12.0, 34.0)
merged_df["MIN"] = np.where(merged_df["MIN"] > 0, merged_df["MIN"], imputed_min).round(1)

merged_df = apply_vegas_and_pace(merged_df, team_pace_stats, vegas_edited_df, blowout_spread=blowout_threshold)

# Sidebar: Injury Scratches with CSV Auto-Detection
st.sidebar.header("3. Injury Status (Scratches)")
all_names = sorted(merged_df["Name"].unique().tolist())

auto_scratches = []
if "Instructions" in merged_df.columns:
    detected = merged_df[
        merged_df["Instructions"].astype(str).str.strip().str.upper().isin(["O", "OUT", "IR", "SUS"])
    ]["Name"].tolist()
    auto_scratches = [p for p in detected if p in all_names]

outs = st.sidebar.multiselect(
    "Mark OUT (Auto-detected from CSV + manual adds)",
    options=all_names,
    default=auto_scratches
)

pool_df = reallocate_vacated_usage(merged_df, outs)

# Sidebar: Quality & Minutes Filters
st.sidebar.header("4. Quality & Minutes Filters")
min_proj_minutes = st.sidebar.slider("Min Expected Minutes", min_value=0.0, max_value=25.0, value=15.0, step=1.0)
pool_df = pool_df[pool_df["Adj_MIN"] >= min_proj_minutes].copy()

if len(pool_df) < 15:
    st.sidebar.warning("⚠️ Min Expected Minutes filter is too strict. Showing top available players.")
    pool_df = reallocate_vacated_usage(merged_df, outs).sort_values(by="Projected_FPTS", ascending=False).head(40)

pool_df = calculate_percentile_ceilings(pool_df)

# Sidebar: Tournament Controls
st.sidebar.header("5. GPP & Portfolio Controls")

target_metric = st.sidebar.selectbox(
    "Optimization Objective",
    options=["Ceiling_90", "Ceiling_75", "Projected_FPTS"],
    format_func=lambda x: {
        "Ceiling_90": "🔥 90th Percentile (Max GPP Upside)",
        "Ceiling_75": "🎯 75th Percentile (Balanced Tournament)",
        "Projected_FPTS": "📊 Median / 50th Percentile (Cash/Floor)"
    }[x]
)

min_studs_req = st.sidebar.number_input("Min Studs ($9,500+ Cap Anchors)", min_value=0, max_value=3, value=1)
max_players_per_game = st.sidebar.slider("Max Players From Any Single Game", min_value=2, max_value=5, value=4, help="Hard cap per game. 3 or 4 prevents single-game cannibalization.")

available_active = sorted(pool_df["Name"].unique())
locks = st.sidebar.multiselect("🔒 Lock Core Players", options=available_active)
fades = st.sidebar.multiselect("❌ Exclude Players", options=[p for p in available_active if p not in locks])

max_exp = st.sidebar.slider(
    "Max Player Exposure across 5 Lineups",
    min_value=0.20, max_value=1.0, value=0.60, step=0.20,
    help="0.60 means a player can appear in at most 3 of the 5 lineups."
)

slate_games = ["None"] + slate_matchups
game_stack = st.sidebar.selectbox("🎯 Target Game Stack", options=slate_games)
stack_players = st.sidebar.slider("Min Players from Stacked Game", 2, 4, 3)

# --- 7. TABS ---
tab_lineups, tab_portfolio, tab_pool = st.tabs(["⚡ 5 Optimized Lineups", "📊 Exposure Breakdown", "📋 Player Pool & Vegas Projections"])

with tab_lineups:
    if st.button("🚀 Generate 5 Tournament Lineups", type="primary"):
        if len(pool_df) < 8:
            st.error("Player pool has fewer than 8 eligible players. Lower the 'Min Expected Minutes' slider in the sidebar.")
        else:
            with st.spinner("Solving multi-lineup linear integer program with SciPy..."):
                five_lineups = optimize_5_lineups_scipy(
                    pool_df=pool_df,
                    locks=locks,
                    excludes=fades,
                    cap=salary_cap,
                    max_exposure=max_exp,
                    target_stack=game_stack,
                    stack_size=stack_players,
                    max_game_players=max_players_per_game,
                    objective_metric=target_metric,
                    min_studs=min_studs_req
                )

            if len(five_lineups) < 5:
                st.warning(f"Solver generated {len(five_lineups)} valid lineups before running out of valid combinations. Try lowering the minimum expected minutes, reducing stud requirements, or relaxing exposure limits.")

            if five_lineups:
                export_rows = []
                for idx, l_df in enumerate(five_lineups):
                    target_sum = l_df[target_metric].sum().round(1)
                    med_sum = l_df["Projected_FPTS"].sum().round(1)
                    sal_sum = l_df["Salary"].sum()
                    st.subheader(f"Lineup #{idx+1} — {target_sum} Target Pts (Median: {med_sum} pts | ${sal_sum:,} Salary)")

                    st.dataframe(
                        l_df[["Roster Slot", "Name", "Position", "TeamAbbrev", "Salary", "Adj_MIN", "Vegas_ITT", "Projected_FPTS", "Ceiling_75", "Ceiling_90", "Clean_Game"]],
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

        meta = pool_df[["Name", "TeamAbbrev", "Position", "Salary", "Vegas_ITT", "Projected_FPTS", "Ceiling_75", "Ceiling_90"]].drop_duplicates()
        portfolio_summary = pd.merge(exp_df, meta, on="Name", how="left")

        st.dataframe(
            portfolio_summary[["Name", "TeamAbbrev", "Position", "Salary", "Vegas_ITT", "Count", "Exposure %", "Projected_FPTS", "Ceiling_75", "Ceiling_90"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Click 'Generate 5 Tournament Lineups' in the first tab to view portfolio distribution.")

with tab_pool:
    st.subheader("Active Player Pool (Vegas ITT, Pace & Ceiling Metrics)")
    st.dataframe(
        pool_df[["Name", "Position", "TeamAbbrev", "Salary", "Adj_MIN", "Vegas_ITT", "Game_Spread", "Env_Mult", "Projected_FPTS", "Ceiling_75", "Ceiling_90", "Value (x/1k)", "Clean_Game"]]
        .sort_values(by=target_metric, ascending=False),
        use_container_width=True,
        hide_index=True
    )
