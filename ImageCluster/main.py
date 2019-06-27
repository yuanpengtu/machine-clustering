from model import ImageCluster
m=ImageCluster(
    base_model='vgg16',#your feature map extractor model
    resorted_img_folder='resorted_data',#the folder for clustered images
    cluster_algo='kmeans',#cluster algorithm
    base_img_folder='data',
    maxK=150,#the max k num is 30, which means ImageCluster calculates every k in range(2,30+1)
)
# calculate the feature maps
# m.get_feature_map(
#     resize_shape=(224,224) # (w,h)  a tuple for resizing the input images to the same shape
# )
# #clustering for feature maps
# m.imagecluster()
#As we can see, 21 may be the best cluster number for this dataset.
#So,we can call the resorted_img function to label the images under different folders
m.resorted_img(
    selected_k_num=100# a int number in range[2,maxK]
)

