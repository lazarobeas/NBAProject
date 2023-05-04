import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats

filepath = "/data/PPApril27wID.csv"
df = pd.read_csv(filepath)
df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)

player_stats = LeagueDashPlayerStats(season="2022-23").get_data_frames()[0]
player_stats_10games = LeagueDashPlayerStats(last_n_games="10").get_data_frames()[0]

prop_type_mapping = {
    'Points': 'PTS',
    'Assists': 'AST',
    'Rebounds': 'REB',
    'Fantasy Score': 'NBA_FANTASY_PTS',
    '3-PT Made': 'FG3M',
    'Free Throws Made': 'FTM',
    'Blocks': 'BLK',
    'Steals': 'STL',
    'Personal Fouls': 'PF',
    'Turnovers': 'TOV',
    'FG Attempted': 'FGA',
    'Games Played': 'GP',
}

prop_totals = {}

for prop_type, api_prop_type in prop_type_mapping.items():
    prop_totals[prop_type] = player_stats[api_prop_type]

# Add prop_totals for the last 10 games
prop_totals_10games = {}
for prop_type, api_prop_type in prop_type_mapping.items():
    prop_totals_10games[prop_type] = player_stats_10games[api_prop_type]

for index, row in df.iterrows():
    player_id = row['ID']
    player_stats_filtered = player_stats.loc[player_stats['PLAYER_ID'] == player_id]
    games_played = player_stats_filtered[prop_type_mapping['Games Played']].sum()

    for prop_type, api_prop_type in prop_type_mapping.items():
        prop_total = player_stats_filtered[api_prop_type].sum()
        prop_per_game = prop_total / games_played

        df.at[index, prop_type + ' Totals'] = prop_total
        df.at[index, prop_type + ' Per Game'] = prop_per_game

    # Update dataframe with last 10 games prop totals
    player_stats_10games_filtered = player_stats_10games.loc[player_stats_10games['PLAYER_ID'] == player_id]
    games_played_10games = player_stats_10games_filtered[prop_type_mapping['Games Played']].sum()

    for prop_type, api_prop_type in prop_type_mapping.items():
        prop_total_10games = player_stats_10games_filtered[api_prop_type].sum()
        prop_per_game_10games = prop_total_10games / games_played_10games

        df.at[index, prop_type + ' Per L10'] = prop_per_game_10games

combined_stats = {
    'PTS+REB+AST': ['Points Per Game', 'Rebounds Per Game', 'Assists Per Game'],
    'PTS+AST': ['Points Per Game', 'Assists Per Game'],
    'PTS+REB': ['Points Per Game', 'Rebounds Per Game'],
    'REB+AST': ['Rebounds Per Game', 'Assists Per Game'],
    'BLKS+STLS': ['Blocks Per Game', 'Steals Per Game'],
}

for combined_stat, components in combined_stats.items():
    df[combined_stat] = df[components].sum(axis=1)

combined_stats_10games = {
    'PTS+REB+AST L10': ['Points Per L10', 'Rebounds Per L10', 'Assists Per L10'],
    'PTS+AST L10': ['Points Per L10', 'Assists Per L10'],
    'PTS+REB L10': ['Points Per L10', 'Rebounds Per L10'],
    'REB+AST L10': ['Rebounds Per L10', 'Assists Per L10'],
    'BLKS+STLS L10': ['Blocks Per L10', 'Steals Per L10'],
}

for combined_stat_10games, components in combined_stats_10games.items():
    df[combined_stat_10games] = df[components].sum(axis=1)

filepath2 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\PPApril27wStats2022-23.csv'


df = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)
df = df.drop(df.columns[df.columns.str.contains('Games Played Per Game', 'Games Played Per Last 10 Games')], axis=1)
df.to_csv(filepath2, index=False)
print(df)
