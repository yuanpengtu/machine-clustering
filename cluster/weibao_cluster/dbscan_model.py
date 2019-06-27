#encoding=utf-8
from sklearn.metrics import calinski_harabaz_score
from weibao_cluster import clu_data_helper
from sklearn.cluster import DBSCAN
import time
X=clu_data_helper.get_sen_vec(90,50)
_,y=clu_data_helper.get_content()
start=time.time()
# for i in [11,12,13,14,15]:
#     for data in [0.9,0.8,1.0]:
y_pred = DBSCAN(eps=0.8,min_samples=12).fit_predict(X)
n_clusters_ = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        # print("eps",data,"min_samples",i,'n_clusters_',n_clusters_,"Calinski-Harabasz Score",calinski_harabaz_score(X, y_pred))
end=time.time()
print(end-start)
print("Calinski-Harabasz Score",calinski_harabaz_score(X, y_pred))