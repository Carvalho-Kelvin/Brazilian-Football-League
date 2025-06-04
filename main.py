import pandas as pd
from loader import load_matches
import re
from gspread_dataframe import get_as_dataframe, set_with_dataframe
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from math import exp, factorial

# Cleaning data and padronizing team names. 
TEAM_MAP = {
    'Atletico MG': 'Atlético-MG',
    'Atlético MG': 'Atlético-MG',
    'Bragantino': 'Red Bull Bragantino',
    'RB Bragantino': 'Red Bull Bragantino',
    'Vasco da Gama': 'Vasco',
    'Vitoria': 'Vitória'
}

# List of teams that we'll analise (teams from Brazilian 2025 Serie A)
SERIE_A_TEAMS = [
    'Atlético-MG', 'Bahia', 'Botafogo', 'Ceará', 'Corinthians',
    'Cruzeiro', 'Flamengo', 'Fluminense', 'Fortaleza', 'Grêmio',
    'Internacional', 'Juventude', 'Mirassol', 'Palmeiras', 'Red Bull Bragantino',
    'Santos', 'São Paulo', 'Sport', 'Vasco', 'Vitória'
]

# List of features that we'll get data and compare with points per game
features = [
    'Poss_pg', 'GF_pg', 'GA_pg', 'SoG_pg', 'SoGA_pg', 'ShAtt_pg', 
    'ShAttA_pg', 'Corners_pg', 'CornersA_pg', 'Saves_pg',
    'SavesA_pg'
]

# Help function to clean date field in our database
def clean_date(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True)
    return df

# Help function to convert our collumns from string to numbers 
def convert_str_to_num(df):
    df = df.copy()
    num_cols = df.columns.drop(['Date', 'Home_team', 'Away_team', 'Comp']) # convert all collumns except these
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    return df

# Help function to convert ball possession to decimal format
def convert_to_decimal(df):
    df = df.copy()
    poss_cols = ['Home_team_poss', 'Away_team_poss']
    df[poss_cols] = df[poss_cols] / 100.0
    return df

# Help function to normalize team names 
def normalize_teams(df):
    df = df.copy()
    for col in ['Home_team', 'Away_team']:
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', x)) # Added this to normalize invisible characters that I found in some teams names
        df[col] = df[col].str.strip()
        df[col] = df[col].replace(TEAM_MAP)
    return df

# Help function to calculate lambda numbers for home and away teams, using for and against stats from dataset
def team_lambda(home, away, stat_for, stat_against):
    h_for = home_rank.loc[home, stat_for]
    h_again = home_rank.loc[home, stat_against]
    a_for = away_rank.loc[away, stat_for]
    a_again = away_rank.loc[away, stat_against]
    return 0.5*(h_for + a_again) + 0.5*(a_for + h_again)

# Help function to calculate poisson distribution (used to calculate percentages for matches outcomes)
def poisson_pmf(k, lam):
    return lam**k * exp(-lam) / factorial(k)

"""
Calculate the possible matches outcomes using lambda for checking the chances for each team score x goals until 12 (choosed because it's a 
high number, rarelly to be scored in football games, which improved our predictions - with low number (6, for example, we had 
probabiblities that combined were over 100%))
"""
def result_probs(lam1, lam2, max_goals=12):
    ph = [poisson_pmf(i, lam1) for i in range(max_goals+1)] # poisson distribution for home team
    pa = [poisson_pmf(j, lam2) for j in range(max_goals+1)] # poisson distribution for away team

    # Probability for home team win
    p_home = sum(ph[i] * pa[j] for i in range(max_goals+1)
                                for j in range(max_goals+1) if i>j)
    p_draw = sum(ph[i] * pa[i] for i in range(max_goals+1)) # Probability for a drawn
    p_away = 1 - p_home - p_draw # Pobability for away team win (simply deduct p_home and p_draw of 1)
    return p_home, p_draw, p_away

"""
Helper function to build team rankings.

Arguments:
    df: Input DataFrame.
    team_list: List of teams.
    home_away: 'all', 'home', or 'away' to specify ranking type.

Returns:
    DataFrame with team ranking statistics.
"""
def _build_ranking(df: pd.DataFrame, team_list: list[str], home_away: str = 'all') -> pd.DataFrame:
   
    # Checks if stats are from home/away matches, putting them in different categories
    # Creating new collumns for our rankings
    if home_away == 'home':
        relevant_df = df[[
            'Home_team',
            'Home_team_poss', 'Away_team_poss',
            'Home_team_goals', 'Away_team_goals',
            'Home_team_sog', 'Away_team_sog',
            'Home_team_shots_attemps', 'Away_team_shots_attemps',
            'Home_team_corners', 'Away_team_corners',
            'Home_team_yellow_cards', 'Away_team_yellow_cards',
            'Home_team_saves', 'Away_team_saves'
        ]].copy()
        relevant_df.columns = [
            'Team',
            'Poss_for', 'Poss_against',
            'Goals_for', 'Goals_against',
            'SoG_for', 'SoG_against',
            'ShAtt_for', 'ShAtt_against',
            'Corners_for', 'Corners_against',
            'Yellow_for', 'Yellow_against',
            'Saves_for', 'Saves_against'
        ]
    elif home_away == 'away':
        relevant_df = df[[
            'Away_team',
            'Away_team_poss', 'Home_team_poss',
            'Away_team_goals', 'Home_team_goals',
            'Away_team_sog', 'Home_team_sog',
            'Away_team_shots_attemps', 'Home_team_shots_attemps',
            'Away_team_corners', 'Home_team_corners',
            'Away_team_yellow_cards', 'Home_team_yellow_cards',
            'Away_team_saves', 'Home_team_saves'
        ]].copy()
        relevant_df.columns = [
            'Team',
            'Poss_for', 'Poss_against',
            'Goals_for', 'Goals_against',
            'SoG_for', 'SoG_against',
            'ShAtt_for', 'ShAtt_against',
            'Corners_for', 'Corners_against',
            'Yellow_for', 'Yellow_against',
            'Saves_for', 'Saves_against'
        ]
    elif home_away == 'all':
        home = df[[
            'Home_team',
            'Home_team_poss', 'Away_team_poss',
            'Home_team_goals', 'Away_team_goals',
            'Home_team_sog', 'Away_team_sog',
            'Home_team_shots_attemps', 'Away_team_shots_attemps',
            'Home_team_corners', 'Away_team_corners',
            'Home_team_yellow_cards', 'Away_team_yellow_cards',
            'Home_team_saves', 'Away_team_saves'
        ]].copy()
        home.columns = [
            'Team',
            'Poss_for', 'Poss_against',
            'Goals_for', 'Goals_against',
            'SoG_for', 'SoG_against',
            'ShAtt_for', 'ShAtt_against',
            'Corners_for', 'Corners_against',
            'Yellow_for', 'Yellow_against',
            'Saves_for', 'Saves_against'
        ]

        away = df[[
            'Away_team',
            'Away_team_poss', 'Home_team_poss',
            'Away_team_goals', 'Home_team_goals',
            'Away_team_sog', 'Home_team_sog',
            'Away_team_shots_attemps', 'Home_team_shots_attemps',
            'Away_team_corners', 'Home_team_corners',
            'Away_team_yellow_cards', 'Home_team_yellow_cards',
            'Away_team_saves', 'Home_team_saves'
        ]].copy()
        away.columns = home.columns  # same names

        relevant_df = pd.concat([home, away], ignore_index=True)
    else:
        raise ValueError("home_away must be 'all', 'home', or 'away'")

    relevant_df['Win'] = (relevant_df['Goals_for'] > relevant_df['Goals_against']).astype(int)
    relevant_df['Draw'] = (relevant_df['Goals_for'] == relevant_df['Goals_against']).astype(int)
    relevant_df['Loss'] = (relevant_df['Goals_for'] < relevant_df['Goals_against']).astype(int)
    relevant_df['Points'] = relevant_df['Win'] * 3 + relevant_df['Draw']

    # Summarizing important stats that we want to highlight for each team
    summary = (
        relevant_df.groupby('Team').agg(
            Games=('Team', 'size'),
            Wins=('Win', 'sum'),
            Draws=('Draw', 'sum'),
            Losses=('Loss', 'sum'),
            Poss=('Poss_for', 'sum'),
            GF=('Goals_for', 'sum'),
            GA=('Goals_against', 'sum'),
            Pts=('Points', 'sum'),
            SoG=('SoG_for', 'sum'),
            SoGA=('SoG_against', 'sum'),
            ShAtt=('ShAtt_for', 'sum'),
            ShAttA=('ShAtt_against', 'sum'),
            Corners=('Corners_for', 'sum'),
            CornersA=('Corners_against', 'sum'),
            Yellow=('Yellow_for', 'sum'),
            YellowA=('Yellow_against', 'sum'),
            Saves=('Saves_for', 'sum'),
            SavesA=('Saves_against', 'sum'),
        )
        .loc[team_list]
        .reset_index()
    )

    # Putting our stats in per game metric
    for col in ['Pts', 'Poss', 'GF', 'GA', 'SoG', 'SoGA', 'ShAtt', 'ShAttA',
                'Corners', 'CornersA', 'Yellow', 'YellowA', 'Saves', 'SavesA']:
        summary[f"{col}_pg"] = (summary[col] / summary['Games']).round(3)

    summary['Pct_GF_Shat'] = (summary['GF'] / summary['ShAtt']).round(3)
    summary['Pct_GF_SoG'] = (summary['GF'] / summary['SoG']).round(3)
    summary['Pct_Sv_SoGAg'] = (summary['Saves'] / summary['SoGA']).round(3)

    return summary

# Returns (rank, home_rank, away_rank).
def build_rankings() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  
    raw   = load_matches() # load matches from dataset using a help function for that (different file for organization)
    # Sequency of cleaning data before using
    clean = clean_date(raw)
    clean = convert_str_to_num(clean)
    clean = convert_to_decimal(clean)
    clean = normalize_teams(clean)

    # Creating 3 different rankings (first for general numbers, second for numbers for home matches of these teams and third for away matches)
    rank = _build_ranking(clean, SERIE_A_TEAMS, 'all')
    home_rank = _build_ranking(clean, SERIE_A_TEAMS, 'home')
    away_rank = _build_ranking(clean, SERIE_A_TEAMS, 'away')
    return rank, home_rank, away_rank, clean

# Returns a DataFrame with ['Team','GF_pg_last5','GA_pg_last5']. Used to estimate if teams are in a good or bad moment.
def compute_last5_stats(df, team_list):
    # 1. Pivot to long form: one row per team–match
    home = df[['Date','Home_team','Home_team_goals','Away_team_goals']].copy()
    home.columns = ['Date','Team','Goals_for','Goals_against']
    
    away = df[['Date','Away_team','Away_team_goals','Home_team_goals']].copy()
    away.columns = ['Date','Team','Goals_for','Goals_against']
    
    long = pd.concat([home, away], ignore_index=True)
    
    # 2. Sort by team and date
    long = long.sort_values(['Team','Date'])
    
    # 3. For each team take its last 5 rows
    last5 = long.groupby('Team').tail(5)
    
    # 4. Aggregate sums and counts
    agg = (
        last5
        .groupby('Team')
        .agg(
            GF_last5    = ('Goals_for',      'sum'),
            GA_last5    = ('Goals_against',  'sum'),
            Games_last5 = ('Goals_for',      'count')
        )
        .reindex(team_list)  # ensure same order, missing teams get NaN
        .fillna(0)
        .reset_index()
    )
    
    # 5. Compute per‑game averages
    agg['GF_pg_last5'] = (agg['GF_last5'] / agg['Games_last5']).round(3)
    agg['GA_pg_last5'] = (agg['GA_last5'] / agg['Games_last5']).round(3)
    
    return agg[['Team','GF_pg_last5','GA_pg_last5']]

# Update rounds table, adding our xGs calculations and stats from teams based on confrontation and venue of that round
def update_round_xgs(wb, sheet_name, overall_rank, home_rank, away_rank, league_stats, last5_stats):
    ws = wb.worksheet(sheet_name)

    fixtures = get_as_dataframe(ws, evaluate_formulas=True)
    fixtures = fixtures.dropna(subset=['Home_team', 'Away_team'])

    df = fixtures.copy()

    # Ensure 'Home_team' is consistent (string and no extra spaces)
    df['Home_team'] = df['Home_team'].astype(str).str.strip()

    # Convert the index of overall_rank to string
    overall_rank.index = overall_rank.index.astype(str)

    # Load feature importances from another file (used to estimate importance for stats to win games)
    try:
        feature_importances = pd.read_csv('feature_importances.csv')
        feature_importances = feature_importances.set_index('Feature')['Importance'].to_dict()
    except FileNotFoundError:
        print("Warning: feature_importances.csv not found. Using default weights.")
        feature_importances = {feature: 1 for feature in features}
    
    def get_weighted_stat(team_rank, stat, default_weight=1):
        return team_rank[stat] * feature_importances.get(stat, default_weight)

    # xG1 for home team - based on general perfomance. Uses Goals For per game from Home Team x Goals Against per game from Away Team to calculate who is favorite to win that game
    df = (
        df
        .merge(overall_rank[['GF_pg']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'GF_pg': 'Home_GF_pg'})
        .merge(overall_rank[['GA_pg']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'GA_pg': 'Away_GA_pg'})
    )
    df['xG1_home'] = get_weighted_stat(df, 'Home_GF_pg') * get_weighted_stat(df, 'Away_GA_pg')

    # xG1 for away team - based on general perfomance. Uses Goals For per game from Away Team x Goals Against per game from Home Team to calculate who is favorite to win that game
    df = (
        df
        .merge(overall_rank[['GF_pg']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'GF_pg': 'Away_GF_pg'})
        .merge(overall_rank[['GA_pg']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'GA_pg': 'Home_GA_pg'})
    )
    df['xG1_away'] = get_weighted_stat(df, 'Away_GF_pg') * get_weighted_stat(df, 'Home_GA_pg') # using weight for stats

    # xG2 (home/away splits) - similar to xG1, but based only on home performance for home teams and away performances for away teams 
    df = (
        df
        .merge(home_rank[['GF_pg', 'GA_pg']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'GF_pg':'Home_GFpg_home', 'GA_pg':'Home_GApg_home'})
        .merge(away_rank[['GF_pg', 'GA_pg']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'GF_pg':'Away_GFpg_away', 'GA_pg':'Away_GApg_away'})
    )
    df['xG2_home'] = get_weighted_stat(df, 'Home_GFpg_home') * get_weighted_stat(df, 'Away_GApg_away') # using weight for stats
    df['xG2_away'] = get_weighted_stat(df, 'Away_GFpg_away') * get_weighted_stat(df, 'Home_GApg_home') # using weight for stats

    # xG3 (shots-on-goal) - calculation of how many goals a team should have per game based on their number of shots on target (leagues avarage as parameter)
    # Also checking how many goals against per game a team should have based on the number of shots on target they allow 
    pct_sog = league_stats['pct_goals_per_sog'] # percentage of league goals for shots on target 
    df = (
        df
        .merge(overall_rank[['SoG_pg', 'SoGA_pg']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'SoG_pg':'Home_SoG_pg', 'SoGA_pg':'Home_SoGA_pg'})
        .merge(overall_rank[['SoG_pg', 'SoGA_pg']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'SoG_pg':'Away_SoG_pg', 'SoGA_pg':'Away_SoGA_pg'})
    )
    df['xG3_home'] = (get_weighted_stat(df, 'Home_SoG_pg') * pct_sog) * (get_weighted_stat(df, 'Away_SoGA_pg') * pct_sog) # using weight for stats and league conversion percentage
    df['xG3_away'] = get_weighted_stat(df, 'Away_SoG_pg') * pct_sog * (get_weighted_stat(df, 'Home_SoGA_pg') * pct_sog) # using weight for stats and league conversion percentage

    # xG4 (last 5 form for home) - checking home team momentum from 5 last games
    df = (
        df
        .merge(last5_stats[['GF_pg_last5']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'GF_pg_last5':'Home_GF5'})
        .merge(last5_stats[['GA_pg_last5']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'GA_pg_last5':'Away_GA5'})
    )
    df['xG4_home'] = get_weighted_stat(df, 'Home_GF5') * get_weighted_stat(df, 'Away_GA5')

    # xG4 (last 5 form for away) - checking home team momentum from 5 last games
    df = (
        df
        .merge(last5_stats[['GF_pg_last5']], left_on='Away_team', right_index=True, how='left')
        .rename(columns={'GF_pg_last5':'Away_GF5'})
        .merge(last5_stats[['GA_pg_last5']], left_on='Home_team', right_index=True, how='left')
        .rename(columns={'GA_pg_last5':'Home_GA5'})
    )
    df['xG4_away'] = get_weighted_stat(df, 'Away_GF5') * get_weighted_stat(df, 'Home_GA5') # using weight for stats

    # giving weights for our xGs. For now, I'm giving more weight for second xG, to bring more importance for venue.
    weights = {
        'xG1_home': 1, 'xG2_home': 2,
        'xG3_home': 1, 'xG4_home': 1
    }
    total_w = sum(weights.values())

    # weighted xG for home
    df['xG_total_home'] = (
        sum(df[col] * w for col, w in weights.items())
        / total_w
    ).round(3)

    # weighted for away
    weights_away = {
        'xG1_away': 1, 'xG2_away': 2,
        'xG3_away': 1, 'xG4_away': 1
    }
    df['xG_total_away'] = (
        sum(df[col] * w for col, w in weights_away.items())
        / total_w
    ).round(3)

    # difference of xG for home and away teams
    df['xG_diff'] = (df['xG_total_home'] - df['xG_total_away']).round(3)

    # outcome probabilities using our result_probs function and our total xG
    df[['P_home_win', 'P_draw', 'P_away_win']] = df.apply(
        lambda r: result_probs(r.xG_total_home, r.xG_total_away),
        axis=1, result_type='expand'
    )

    # prepare output
    out = fixtures.copy().reset_index(drop=True)
    for col in [
        'xG1_home','xG1_away',
        'xG2_home','xG2_away',
        'xG3_home','xG3_away',
        'xG4_home','xG4_away',
        'xG_total_home','xG_total_away','xG_diff',
        'P_home_win', 'P_draw', 'P_away_win',
        'P_goals_over2.5', 'P_goals_under2.5',
        'P_corn_over9.5', 'P_corn_under9.5',
        'P_corn_over10.5', 'P_corn_under10.5',
        'P_yellow_over4.5', 'P_yellow_under4.5',
        'P_yellow_over5.5', 'P_yellow_under5.5',
        'P_yellow_over6.5', 'P_yellow_under6.5'
    ]:
        out[col] = df[col].round(3)

    set_with_dataframe(ws, out)

# Function to export our output
def export_sheet(wb, df, sheet_name):
    try:
        wb.del_worksheet(wb.worksheet(sheet_name))
    except gspread.exceptions.WorksheetNotFound:
        pass

    ws = wb.add_worksheet(
        title=sheet_name,
        rows=str(len(df) + 1),
        cols=str(len(df.columns))
    )
    set_with_dataframe(ws, df, include_index=True)


if __name__ == "__main__":
    # Using Google API to load our dataset
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name("brazilian-football.json", scope)
    client = gspread.authorize(creds)
    sheet_url = "https://docs.google.com/spreadsheets/d/1ZKm79JbSoFn7ljR7fGQci4uVAx2p5sMXel6hD-ZgQHs"
    wb = client.open_by_url(sheet_url)

    rank, home_rank, away_rank, clean = build_rankings()
    rank = rank.set_index('Team')	
    home_rank = home_rank.set_index('Team')
    away_rank = away_rank.set_index('Team')
    total_goals = clean['Home_team_goals'].sum() + clean['Away_team_goals'].sum()
    total_shatt = clean['Home_team_shots_attemps'].sum() + clean['Away_team_shots_attemps'].sum()
    total_sog = clean['Home_team_sog'].sum() + clean['Away_team_sog'].sum()
    total_saves = clean['Home_team_saves'].sum() + clean['Away_team_saves'].sum()

    # % goals per shot attempt (general)
    pct_goals_shatt = total_goals / total_shatt

    # % goals per shot‑on‑goal (general)
    pct_goals_sog = total_goals / total_sog

    # % saves per shot‑on‑goal against (general)
    # note: total_sog used above is total SoG_for both sides = total SoG_against too
    pct_saves_sogag = total_saves / total_sog

    league_stats = {
        'pct_goals_per_shatt': pct_goals_shatt,
        'pct_goals_per_sog':   pct_goals_sog,
        'pct_saves_per_sogag': pct_saves_sogag
}

    last5_df = compute_last5_stats(clean, SERIE_A_TEAMS)
    last5_df = last5_df.set_index('Team')

    # average goals per match in Série A only
    serie_a = clean[clean['Comp'].str.contains('Serie A', case=False)]
    avg_goals_per_match = ((serie_a['Home_team_goals'] + serie_a['Away_team_goals']).mean())
    avg_corner_per_match = ((serie_a['Home_team_corners'] + serie_a['Away_team_corners']).mean())
    avg_cards_per_match = ((serie_a['Home_team_yellow_cards'] + serie_a['Away_team_yellow_cards']).mean())

    # printing league stats                      
    print(f"League‑wide % goals/shot att:    {pct_goals_shatt:.3f}")
    print(f"League‑wide % goals/SoG:         {pct_goals_sog:.3f}")
    print(f"League‑wide % saves/SoG against: {pct_saves_sogag:.3f}")
    print(f"Avg goals per Série A match:     {avg_goals_per_match:.3f}")
    print(f"Avg corners per Série A match:     {avg_corner_per_match:.3f}")
    print(f"Avg cards per Série A match:     {avg_cards_per_match:.3f}")

    # exporting 3 new sheets for our different rankings
    export_sheet(wb, rank, 'Ranking')
    export_sheet(wb, home_rank, 'Ranking_Home')
    export_sheet(wb, away_rank, 'Ranking_Away')

    # update the current round expectations
    update_round_xgs(
        wb,
        sheet_name    = "Round11",
        overall_rank  = rank,
        home_rank     = home_rank,
        away_rank     = away_rank,
        league_stats  = league_stats,
        last5_stats   = last5_df
)