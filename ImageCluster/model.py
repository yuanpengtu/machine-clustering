from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.xception import Xception
from keras.preprocessing import image
import os, pandas as pd
import numpy as np
from sklearn.cluster import KMeans
class Model:
    def __init__(self,model_name):
        self.model=model_name
    def build_model(self):
        if self.model=='Xception' or self.model=='xception':
            return Xception(include_top=False,pooling='avg',weights='imagenet'),'Xception'
        elif self.model=='DenseNet' or self.model=='densenet':
            return DenseNet121(include_top=False,pooling='avg',weights='imagenet'),'DenseNet'
        elif self.model=='ResNet' or self.model=='resnet':
            return ResNet50(include_top=False,pooling='avg',weights='imagenet'),'ResNet'
        elif self.model=='VGG16' or self.model=='vgg16':
            return VGG16(include_top=False,pooling='avg',weights='imagenet'),'VGG16'
        return VGG19(include_top=False,pooling='avg',weights='imagenet'),'VGG19'

def get_features(img_folder,Model):
    fp1=pd.DataFrame(columns=['filename','feature'])
    fp2 = pd.DataFrame(columns=['filename', 'feature'])
    for i in os.listdir(img_folder):#stage1 stage2
        next_folder=os.path.join(img_folder,i)
        for j in os.listdir(next_folder):#Belt st1...
            next_folder_plus=os.path.join(next_folder,j)
            for h in os.listdir(next_folder_plus):
                path=os.path.join(next_folder_plus,h)
                print(path)
                img=image.load_img(path,target_size=(224,224))
                x=image.img_to_array(img)
                x=np.expand_dims(x,axis=0)
                features=Model[0].predict(x)[0]
                fp1=fp1.append({'filename':path,'feature':features},ignore_index=True)

                img = image.load_img(path)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                features = Model[0].predict(x)[0]
                fp2 = fp2.append({'filename': path, 'feature': features}, ignore_index=True)

                print(path+' finished!!')
    fp1.to_csv('data/{}_features_fixed_size.csv'.format(Model[1]),index=False)
    fp2.to_csv('data/{}_features_default_size.csv'.format(Model[1]),index=False)
    print('ALL DONE!')

class ImageCluster(object):
    def __init__(self,base_img_folder,resorted_img_folder,
                 cluster_algo='kmeans',base_model='vgg16',k=None,maxK=None,csv_file_path=None):
        self.base_model, self.base_model_name = Model(model_name=base_model).build_model()
        self.csv_file_path=csv_file_path
        self.cluster_algo=cluster_algo
        self.k=k
        self.maxK=maxK

        self.base_img_folder=base_img_folder
        self.resorted_img_folder=resorted_img_folder
    def get_feature_map(self,resize_shape=None):
        """
        You can fix the code as your actual situation and environment!!!
        设置图片文件夹一定是3层的，
        base_model_folder
            ----label
                ----xxxxxxxx.jpg
        :param resize_shape:
        :return:
        """
        img_path_all=[]
        f=pd.DataFrame(columns=['filename','feature'])
        if resize_shape==None:
            if os.path.isdir(self.base_img_folder):
                for i in os.listdir(self.base_img_folder):
                    next_path=os.path.join(os.path.join(self.base_img_folder,i))
                    if os.path.isdir(next_path):
                        for j in os.listdir(next_path):
                            last_path = os.path.join(next_path, j)
                            for w in os.listdir(last_path):
                                img_path = os.path.join(last_path, w)
                                img_path_all.append(img_path)
                                img = image.load_img(img_path, target_size=resize_shape)
                                x = image.img_to_array(img)
                                x = np.expand_dims(x, axis=0)
                                features = self.base_model.predict(x)[0]
                                f = f.append({'filename': img_path, 'feature': features}, ignore_index=True)
                    else:
                        pass

            else:
                raise ValueError('the base image folder is wrong! PLZ check it out')
        else:
            if os.path.isdir(self.base_img_folder):
                for i in os.listdir(self.base_img_folder):
                    next_path=os.path.join(os.path.join(self.base_img_folder,i))
                    if os.path.isdir(next_path):
                        for j in os.listdir(next_path):
                            last_path = os.path.join(next_path, j)
                            for w in os.listdir(last_path):
                                img_path=os.path.join(last_path,w)
                                img_path_all.append(img_path)
                                img = image.load_img(img_path, target_size=resize_shape)
                                x = image.img_to_array(img)
                                x = np.expand_dims(x, axis=0)
                                features = self.base_model.predict(x)[0]
                                f = f.append({'filename': img_path, 'feature': features}, ignore_index=True)
                                print(img_path,' extracted features')


                    else:
                        pass

            else:
                raise ValueError('the base image folder is wrong! PLZ check it out')

        if len(img_path_all)==0:
            raise ValueError('image loading fails,please check you image path!')
        else:
            print('Have got the feature map for each image')
            f.to_csv('output/base_model_{}_feature_maps.csv'.format(self.base_model_name))
            print('output/base_model_{}_feature_maps.csv has finished!'.format(self.base_model_name))


    def kmeans(self):
        x=[]
        if self.csv_file_path == None:
            self.csv_file = pd.read_csv('output/base_model_{}_feature_maps.csv'.format(self.base_model_name))
        else:
            self.csv_file = pd.read_csv(self.csv_file_path)
        for i in self.csv_file['feature']:
            x.append([float(t) for t in i.strip('[').strip(']').split(' ')])
        x=np.array(x)
        if os.path.exists('output'):
            pass
        else:
            os.mkdir('output')

        if os.path.exists('matplot'):
            pass
        else:
            os.mkdir('matplot')

        def func(k):
            model = KMeans(n_clusters=k, init='k-means++')
            model.fit(x)
            # print('cluster_center', model.cluster_centers_)
            f = pd.DataFrame(columns=['filename', 'label'])
            f['filename'] = self.csv_file['filename']
            f['label'] = model.labels_
            f.to_csv('output/base_model_{}_cluster_kmeans_{}.csv'.format(self.base_model_name,str(k)))
            return model.inertia_

        if self.k==None:
            sse=[]
            for k in range(51,self.maxK+1):
                sse.append(func(k))
            import matplotlib.pyplot as plt
            plt.plot(range(2,self.maxK+1),sse,marker='o')
            plt.xlabel('number of K(cluster)')
            plt.ylabel('SSE Value for each K')
            plt.title('KMeans for ImageCluster')
            plt.savefig('matplot/base_model_{}_KMeans_maxK_{}.png'.format(self.base_model_name,str(self.maxK)))
            plt.show()
        else:
            func(self.k)
    def imagecluster(self):
        if self.cluster_algo.lower()=='kmeans':
            self.kmeans()
        else:
            print('no existing cluster algorithm')
            return

    def resorted_img(self,selected_k_num):
        import shutil
        if os.path.exists(self.resorted_img_folder):
            pass
        else:
            os.mkdir(self.resorted_img_folder)

        resorted_csv=pd.read_csv('output/base_model_{}_cluster_kmeans_{}.csv'.format(self.base_model_name,str(selected_k_num)))
        for i in resorted_csv.index:
            filename=resorted_csv.loc[i,'filename']
            label=resorted_csv.loc[i,'label']
            if os.path.exists(os.path.join(self.resorted_img_folder,str(label))):
                pass
            else:
                os.mkdir(os.path.join(self.resorted_img_folder,str(label)))
            shutil.copy(filename,os.path.join(self.resorted_img_folder,str(label)))
            print(os.path.join(self.resorted_img_folder,str(label))+'\\'+filename+' Copied！')
