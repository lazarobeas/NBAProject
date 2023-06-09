import pandas as pd

# filepath0304 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits03-04.csv'
# filepath0405 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits04-05.csv'
# filepath0506 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits05-06.csv'
# filepath0607 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits06-07.csv'
# filepath0708 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits07-08.csv'
# filepath0809 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits08-09.csv'
# filepath0910 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits09-10.csv'
# filepath1011 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits10-11.csv'
filepath1112 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits11-12.csv'
filepath1213 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits12-13.csv'
filepath1314 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits13-14.csv'
filepath1415 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits14-15.csv'
filepath1516 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits15-16.csv'
filepath1617 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits16-17.csv'
filepath1718 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits17-18.csv'
filepath1819 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits18-19.csv'
filepath1920 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits19-20.csv'
filepath2021 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits20-21.csv'
filepath2122 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits21-22.csv'
filepath2223 = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameSplits22-23.csv'

# df19 = pd.read_csv(filepath0304)
# df18 = pd.read_csv(filepath0405)
# df17 = pd.read_csv(filepath0506)
# df16 = pd.read_csv(filepath0607)
# df15 = pd.read_csv(filepath0708)
# df14 = pd.read_csv(filepath0809)
# df9 = pd.read_csv(filepath0910)
# df10 = pd.read_csv(filepath1011)
# df11 = pd.read_csv(filepath1112)
# df12 = pd.read_csv(filepath1213)
# df13 = pd.read_csv(filepath1314)
df0 = pd.read_csv(filepath1718)
# df1 = pd.read_csv(filepath1617)
df2 = pd.read_csv(filepath1920)
df3 = pd.read_csv(filepath2021)
df4 = pd.read_csv(filepath2122)
df5 = pd.read_csv(filepath2223)
# df6 = pd.read_csv(filepath1415)
# df7 = pd.read_csv(filepath1516)
df8 = pd.read_csv(filepath1819)

combinedDF = pd.concat([df0,df8,df2,df3,df4,df5], ignore_index=True)

filepathNew = 'C:\\Users\\Lazaro B\\Documents\\GitHub\\NBAProject\\data\\JamalMurray\\JamalMurrayGameEntireCareer.csv'
combinedDF = combinedDF.drop(combinedDF.columns[combinedDF.columns.str.contains('unnamed', case=False)], axis=1)
combinedDF.sort_values("GAME_DATE")
print(combinedDF)
combinedDF.to_csv(filepathNew)
