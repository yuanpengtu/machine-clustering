#encoding=utf-8
import numpy as np
import tensorflow as tf
#实现位置编码
def position_encoding(sentence_size, embedding_size):
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    encoding[:, -1] = 1.0
    return np.transpose(encoding)
with tf.Session() as sess:
    #这是模拟通过embedding层获得的二维的词向量
    data = np.random.randn(9, 20)
    zero_data=np.zeros((1,20))
    data=np.concatenate((data,zero_data),axis=0)
    #通过位置编码获得句向量
    a=position_encoding(10,20)
    print(data.shape)
    print(a.shape)
    result=tf.reduce_sum(data*a,1)
    print(sess.run(result))
