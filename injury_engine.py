import pandas as pd
import numpy as np
import requests
from io import StringIO

# --- 1. LIVE INJURY REPORT PARSER ---
def fetch_live_injuries():
    """
    Fetches the latest official injury feed.
    Pulls structured injury statuses (Out, Doubtful, Questionable, Probable).
    """
    try:
        # NBA publishes regular updates. We can query live status feeds or rapid sports endpoints
        # For an automated zero-key feed, we can scrape the CBS / ESPN injury aggregator table:
        url = "https://www.cbssports.com/nba/injuries/"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(StringIO(res.text))
        
        injury_records = []
        for t in tables:
            # Columns usually: [Player, Position, Updated, Injury, Status]
            if "Player" in t.columns and "Status" in t.columns:
                for _, row in t.iterrows():
                    injury_records.append({
                        "Player": str(row["Player"]).strip(),
                        "Position": str(row["Position"]).strip() if "Position" in row else "",
                        "Status": str(row["Status"]).strip().lower(),
                        "Injury": str(row["Injury"]).strip() if "Injury" in row else ""
                    })
        injury_df = pd.DataFrame(injury_records)
        return injury_df
    except Exception as e:
        # If offline or scraping blocked, return empty structure
        return pd.DataFrame(columns=["Player", "Position", "Status", "Injury"])


# --- 2. USAGE & MINUTES REALLOCATION LOGIC ---
def apply_injury_usage_bump(player_pool_df, injury_df=None, manual_outs=None):
    """
    Reallocates vacated minutes and usage from OUT players to remaining teammates.
    
    player_pool_df requires columns:
    ['Name', 'TeamAbbrev', 'Position', 'Salary', 'MIN', 'Projected_FPTS']
    """
    df = player_pool_df.copy()
    
    # Add default usage rate estimate if not present (USG% roughly correlates to Salary / Proj)
    if "USG%" not in df.columns:
        # Estimate USG% baseline: 15% (bench/role) to 34% (elite stars)
        df["USG%"] = np.clip(15.0 + (df["Salary"] - 3500) * (18.0 / 8500), 12.0, 36.0)
    
    # Compute baseline Fantasy Points Per Minute (FPPM)
    df["FPPM"] = np.where(df["MIN"] > 0, df["Projected_FPTS"] / df["MIN"], df["Salary"] / 24000.0)
    
    # Identify OUT players
    out_players = set()
    if manual_outs:
        out_players.update(manual_outs)
        
    if injury_df is not None and not injury_df.empty:
        # Match confirmed "out" statuses
        matched_outs = injury_df[injury_df["Status"].str.contains("out|doubtful|expected to miss", case=False, na=False)]["Player"].tolist()
        out_players.update(matched_outs)

    df["Is_Out"] = df["Name"].isin(out_players)
    
    # Process team by team
    adjusted_records = []
    
    for team, group in df.groupby("TeamAbbrev"):
        outs_in_team = group[group["Is_Out"]]
        actives_in_team = group[~group["Is_Out"]].copy()
        
        # If no one is out on this team, keep original values
        if outs_in_team.empty:
            adjusted_records.append(group)
            continue
            
        # Sum vacated volume
        total_vacated_min = outs_in_team["MIN"].sum()
        total_vacated_usg = outs_in_team["USG%"].sum()
        
        # Zero out the OUT players
        outs_in_team = outs_in_team.copy()
        outs_in_team["Projected_FPTS"] = 0.0
        outs_in_team["New_MIN"] = 0.0
        outs_in_team["New_USG%"] = 0.0
        outs_in_team["Usage_Bump_%"] = 0.0
        
        # A. Distribute Minutes:
        # Starters / key rotational players absorb vacated minutes up to a realistic cap (e.g., 36 min)
        min_headroom = np.maximum(36.0 - actives_in_team["MIN"], 0)
        if min_headroom.sum() > 0:
            min_distribution_weight = min_headroom / min_headroom.sum()
            actives_in_team["New_MIN"] = np.minimum(
                actives_in_team["MIN"] + (total_vacated_min * min_distribution_weight * 0.75), 
                38.0
            ).round(1)
        else:
            actives_in_team["New_MIN"] = actives_in_team["MIN"]

        # B. Distribute Usage Rate:
        # Higher-usage players absorb more of the vacant ball-handling duties
        usg_weights = actives_in_team["USG%"] / actives_in_team["USG%"].sum()
        # Scale factor (not all usage is absorbed by key players; some goes to team pace drop)
        absorbed_usg = total_vacated_usg * 0.70
        
        actives_in_team["New_USG%"] = (actives_in_team["USG%"] + (absorbed_usg * usg_weights)).round(2)
        
        # C. Calculate Production Shift (FPPM increases by ~0.75% per 1% USG increase)
        usg_delta_pct = (actives_in_team["New_USG%"] - actives_in_team["USG%"]) / actives_in_team["USG%"]
        new_fppm = actives_in_team["FPPM"] * (1.0 + (usg_delta_pct * 0.75))
        
        actives_in_team["Usage_Bump_%"] = (usg_delta_pct * 100).round(1)
        actives_in_team["Projected_FPTS"] = (actives_in_team["New_MIN"] * new_fppm).round(2)
        
        adjusted_records.append(pd.concat([actives_in_team, outs_in_team]))

    result_df = pd.concat(adjusted_records, ignore_index=True)
    result_df["Value (x/1k)"] = (result_df["Projected_FPTS"] / (result_df["Salary"] / 1000)).round(2)
    return result_df
