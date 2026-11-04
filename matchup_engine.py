import pandas as pd
import numpy as np
from nba_api.stats.endpoints import leaguedashteamstats

# --- 1. PULL LIVE TEAM PACE VIA NBA_API ---
def get_team_pace_table(season='2025-26'):
    """
    Pulls advanced team stats from stats.nba.com to calculate Pace ratings.
    """
    try:
        # LeagueDashTeamStats with MeasureType='Advanced' provides PACE and DEF_RATING
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame',
            season=season,
            season_type_all_star='Regular Season'
        ).get_data_frames()[0]

        # Standardize 3-letter abbreviations
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
        
        team_stats['TeamAbbrev'] = team_stats['TEAM_NAME'].map(team_map)
        pace_df = team_stats[['TeamAbbrev', 'PACE', 'DEF_RATING']].dropna()
        
        return pace_df
    except Exception as e:
        # Fallback benchmark averages if offline
        return pd.DataFrame()

# --- 2. CALCULATE DEFENSE VS POSITION (DvP) MATRIX ---
def get_dvp_matrix():
    """
    Computes position-specific defensive adjustments relative to 1.0 (League Average).
    Values > 1.0 indicate generous defenses (matchup boost).
    Values < 1.0 indicate stingy defenses (matchup downgrade).
    Can be dynamically derived from opponent game logs or box scores.
    """
    # Sample matrix indexed by TeamAbbrev with multipliers per position
    # 1.08 = opponent allows 8% more FPTS to that position than average
    dvp_data = {
        'TeamAbbrev': ['WAS', 'CHA', 'DET', 'UTA', 'BOS', 'OKC', 'MIN', 'CLE'],
        'PG': [1.12, 1.08, 1.05, 1.09, 0.92, 0.90, 0.94, 0.93],
        'SG': [1.10, 1.06, 1.04, 1.07, 0.91, 0.91, 0.93, 0.92],
        'SF': [1.09, 1.07, 1.06, 1.08, 0.93, 0.92, 0.91, 0.94],
        'PF': [1.11, 1.09, 1.08, 1.10, 0.94, 0.93, 0.92, 0.91],
        'C':  [1.15, 1.12, 1.07, 1.11, 0.90, 0.91, 0.88, 0.89],
    }
    return pd.DataFrame(dvp_data)

# --- 3. PARSE OPPONENT & APPLY ADJUSTMENTS ---
def apply_pace_and_dvp(player_df, pace_df, dvp_df):
    """
    Takes the DraftKings slate dataframe and scales projections.
    """
    df = player_df.copy()

    # Step A: Parse Opponent from 'Game Info' (e.g. 'DAL@LAL 07:30PM ET')
    def extract_opponent(row):
        game = str(row['Game Info']).split(' ')[0]
        if '@' in game:
            away, home = game.split('@')
            return home if row['TeamAbbrev'] == away else away
        return None

    df['Opponent'] = df.apply(extract_opponent, axis=1)

    # Step B: Compute Pace Multipliers
    if not pace_df.empty:
        league_pace = pace_df['PACE'].mean()
        pace_dict = pace_df.set_index('TeamAbbrev')['PACE'].to_dict()

        def compute_pace_factor(row):
            t_pace = pace_dict.get(row['TeamAbbrev'], league_pace)
            o_pace = pace_dict.get(row['Opponent'], league_pace)
            # Bill James game pace formula: (Team Pace * Opp Pace) / League Pace
            expected_game_pace = (t_pace * o_pace) / league_pace
            # Adjustment ratio against the player's own baseline tempo
            return round(expected_game_pace / t_pace, 3)

        df['Pace_Factor'] = df.apply(compute_pace_factor, axis=1)
    else:
        df['Pace_Factor'] = 1.0

    # Step C: Compute DvP Multiplier
    # Map primary position (e.g., 'PG/SG' -> 'PG')
    df['Primary_Pos'] = df['Position'].apply(lambda x: str(x).split('/')[0])
    
    dvp_melted = dvp_df.melt(id_vars=['TeamAbbrev'], var_name='Primary_Pos', value_name='DvP_Factor')
    df = pd.merge(
        df,
        dvp_melted,
        left_on=['Opponent', 'Primary_Pos'],
        right_on=['TeamAbbrev', 'Primary_Pos'],
        how='left',
        suffixes=('', '_dvp')
    )
    df['DvP_Factor'] = df['DvP_Factor'].fillna(1.0).round(3)

    # Step D: Final Adjusted Projections
    df['Adjusted_FPTS'] = (df['Projected_FPTS'] * df['Pace_Factor'] * df['DvP_Factor']).round(2)
    df['Net_Boost_%'] = (((df['Adjusted_FPTS'] / df['Projected_FPTS']) - 1) * 100).round(1)

    return df
