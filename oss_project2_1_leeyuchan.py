import pandas as pd

data_df = pd.read_csv('C:/Users/yucha/Downloads/2019_kbo_for_kaggle_v2.csv')

data = pd.DataFrame(data_df)

# 첫번째 문제
for i in range(2015,2019):
    data_1 = data[data['year'] == i]
    record1_data = ['H', 'avg', 'HR', 'OBP']
    for r1 in record1_data:
        Data = data_1.sort_values(by=[r1], ascending=False)
        print("print the top 10 player in ", r1, "at", i, ":\n", Data[['batter_name', r1]].head(10),'\n')

# 두번째 문제
data_2 = data[data['year'] == 2018]
pos_data = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

for pos in pos_data:
    war_data = data_2[data_2['cp'] == pos].sort_values(by=['war'], ascending=False).head(1)
    print("Highest war player by", pos, "in 2018:\n", war_data[['batter_name','war']],'\n')

# 세번째 문제
data_3 = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
corr_matrix = data_3.corr()
corr_matrix.name = 'highest correlation with salary'
print(corr_matrix['salary'].drop('salary').sort_values(ascending=False).head(1))
