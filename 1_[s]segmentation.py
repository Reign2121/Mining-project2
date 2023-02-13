import numpy as np
import pandas as pd
gn = pd.read_csv("공릉동_상권_.csv",encoding = 'cp949')
gn.info()

gn_drop = gn.iloc[:,1:8]
gn_drop = gn_drop.drop('분기당_매출_건수', axis =1) #회귀분석에서 레이블로 사용할 카드 이용건수 변수 제외

gn_drop.columns

#표준화(단위 맞추기)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(gn_drop)
scaled_data

#* 목표1. 각 데이터별 군집분석된 소속 cluster를 도출하기
#* 목표2. 각 cluster별 중심값을 표준화되기 이전 형태로 도출하고 군집별 특징을 파악하기
#* 목표3. n_cluster를 1~10까지 바꿔가면서, inertia기준 적절한 n_cluster를 찾기

##목표1. 각 데이터별 군집분석된 소속 cluster를 도출하기

from sklearn.cluster import KMeans

km = KMeans(n_clusters = 4, random_state = 0) 
km.fit(scaled_data)   
cluster_result = km.predict(scaled_data)

print("군집번호:", cluster_result)
print(km.labels_)
print(np.unique(cluster_result, return_counts = True))

gn["cluster_#"] = cluster_result
gn['cluster_#'].head(50) #10개만 보여주기

gn.value_counts('cluster_#')

##목표2. 각 cluster별 중심값을 표준화되기 이전 형태로 도출하고 군집별 특징을 파악하기
centroid = km.cluster_centers_
original_centroid = scaler.inverse_transform(centroid)

print("원본 군집중심: \n", original_centroid) #표준화 이전 형태로 변환/ 특징 도출

centroid_df = pd.DataFrame(original_centroid, columns = gn_drop.columns)
df_mean = gn_drop.mean()

centroid_df.loc['Average']=df_mean 
display(centroid_df) #군집 중심 평균값 반환
pd.options.display.float_format = '{:.5f}'.format

gn.groupby(['cluster_#', '서비스_업종_코드_명']).size()

#목표3. n_cluster를 1~10까지 바꿔가면서, inertia기준 적절한 n_cluster를 찾기

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

k_num = range(1,10)
inertias = []
#최적의 k 수 찾기
for k in k_num: 
  km_model = KMeans(n_clusters = k, random_state = 0)
  km_model.fit(scaled_data)
  inertias.append(km_model.inertia_) 

plt.plot(k_num, inertias,'-o')
plt.xlabel("# of clusters K")
plt.ylabel("SSE")
plt.show()