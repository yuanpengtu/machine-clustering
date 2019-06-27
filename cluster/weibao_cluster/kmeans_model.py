#encoding=utf-8
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score
from weibao_cluster import clu_data_helper
import time
def get_right_num(y,y_pred):
    num=0
    for i,j in zip(y,y_pred):
        if i==j:
            num=num+1
    return num
X=clu_data_helper.get_sen_vec(90,50)
_,y=clu_data_helper.get_content()
# start=time.time()
for i in [2,3,4,5]:
    y_pred=KMeans(n_clusters=i,algorithm='elkan').fit_predict(X)
    print("Calinski-Harabasz Score", calinski_harabaz_score(X, y_pred))
# end=time.time()
# print(end-start)
# print("Calinski-Harabasz Score",calinski_harabaz_score(X, y_pred))
# print(get_right_num(y,y_pred))