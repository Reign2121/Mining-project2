import numpy as np
import pandas as pd
gn = pd.read_csv("공릉동_상권_.csv",encoding = 'cp949')
gn.info()

##군집분석 결과#
gn_drop = gn.iloc[:,1:8]
gn_drop = gn_drop.drop('분기당_매출_건수', axis =1) #회귀분석에서 레이블로 사용할 카드 이용건수 변수 제외
#표준화(단위 맞추기)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(gn_drop)

from sklearn.cluster import KMeans
km = KMeans(n_clusters = 4, random_state = 0) 
km.fit(scaled_data)   
cluster_result = km.predict(scaled_data)
#print("군집번호:", cluster_result)
#print(km.labels_)
#print(np.unique(cluster_result, return_counts = True)) 
gn["cluster"] = cluster_result

gn['cluster'].value_counts()

gn = pd.get_dummies(gn)
gn.columns

#군집별 데이터 셋 분리
c0 = gn[gn['cluster'] == 0]
c1 = gn[gn['cluster'] == 1]
c2 = gn[gn['cluster'] == 2]
c3 = gn[gn['cluster'] == 3]


"""##Target1 
- 인원 수가 가장 많은 군집 (상권의 주 소비자) : cluster 1
"""

y1 = c1['분기당_매출_건수'] #label
#x1 = c1.drop(['분기당_매출_건수', 'cluster'], axis = 1)
x1 = c1.iloc[:,9:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x1,y1, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
np.set_printoptions(suppress=True, precision=3)
print("회귀계수:", lr_model.coef_, "상수항:", lr_model.intercept_)

y_predict = lr_model.predict(X_test)
sum(y_predict)

np.set_printoptions(suppress=True, precision=3)
import statsmodels.api as sm 
results = sm.OLS(y1, sm.add_constant(x1)).fit() 
results.summary()

list1 = [1,1,1,1,1,1,1,1,1,1,1,1]
coef1 = pd.DataFrame({'군집 번호':list1,'업종': x1.columns,'영향력':lr_model.coef_})
coef1

y_predict = lr_model.predict(X_test)
print({"예상매출건수":sum(y_predict)})

"""##Target2
- 매출이 가장 높은 군집 : cluster 3
"""

y3 = c3['분기당_매출_건수'] #label
#x1 = c1.drop(['분기당_매출_건수', 'cluster'], axis = 1)
x3 = c3.iloc[:,9:]

#표본이 너무 작기 때문에, 따로 데이터 셋 분리x

lr_model2 = LinearRegression()
lr_model2.fit(x3, y3)
np.set_printoptions(suppress=True, precision=3)
print("회귀계수:", lr_model2.coef_, "상수항:", lr_model2.intercept_)

results2 = sm.OLS(y3, sm.add_constant(x3)).fit() 
results.summary()

list2 = [3,3,3,3,3,3,3,3,3,3,3,3]
coef2 = pd.DataFrame({'군집 번호':list2,'업종': x3.columns,'영향력':lr_model2.coef_})
coef2

result = pd.concat([coef1,coef2], axis =1)
result

y_predict2 = lr_model2.predict(x3)
print({"예상매출건수":sum(y_predict2)})
