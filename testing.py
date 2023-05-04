from nba_api.stats.static import players
import pandas as pd
from nba_api.stats.endpoints import PlayerDashboardByGameSplits, PlayerGameLog


capelaid = players.find_players_by_full_name("Robert Williams III")[0]['id']


player_gamelog = PlayerGameLog(player_id=capelaid, season='2022-23').get_data_frames()[0]

column = ['Game_ID', 'GAME_DATE', 'MATCHUP', 'PTS', 'AST', 'STL', 'REB', 'TOV', 'FG3M', 'FG3A', 'BLK',]
capelaGameSplits = player_gamelog[column]

# Show the resulting dataframe
print(capelaGameSplits)

df = pd.DataFrame(capelaGameSplits)
df.to_csv('C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\RobertWilliamsIII\\RobertWilliamsIIIGameSplits22-23.csv')

# df = df.drop(columns=['GAME_DATE', 'MATCHUP']) # Modify this line
#
# # Add new columns
# df['PTS+REB'] = df['PTS'] + df['REB']
# df['PTS+AST'] = df['PTS'] + df['AST']
# df['PTS+REB+AST'] = df['PTS'] + df['REB'] + df['AST']
# df['REB+AST'] = df['REB'] + df['AST']
# df['BLKS+STLS'] = df['BLK'] + df['STL']
#
# print(df)
#
# df.to_csv('C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\')