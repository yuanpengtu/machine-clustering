#encoding=utf-8
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score
import pandas as pd
from weibao_cluster import mul_kmeans_data_helper
# _,content_list=mul_kmeans_data_helper.get_content()
# X=mul_kmeans_data_helper.get_sen_vec(200,50)
# y_pred=KMeans(n_clusters=7,algorithm='elkan').fit_predict(X)
# result=pd.DataFrame({
#     'content':content_list,
#     'label':y_pred
# })
# result.to_csv('pro_data/cluster_content.csv',index=False)
df=pd.read_csv('pro_data/cluster_content.csv',encoding='utf-8')
df.sort_values(by='label',inplace=True)
df.to_csv('pro_data/gbk_cluster_content.csv',index=False,encoding='GBK')
