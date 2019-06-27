#encoding=utf-8
import jieba
import numpy as np
from gensim.models import word2vec
import pandas as pd
def get_content():
    df = pd.read_csv('data/new_content.csv',encoding='GBK')
    content_list = list(df['文本数据'])[5000:20000]
    jieba_cut_list=[jieba.lcut(content) for content in content_list]
    return jieba_cut_list,content_list
#实现位置编码
def position_encoding(sentence_size, embedding_size):
    #sentence_size句子的长度
    #embedding_size每个词的维度
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_size+1)/2) * (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    encoding[:, -1] = 1.0
    return np.transpose(encoding)
#用于返回每个句子的向量
def get_sen_vec(sentence_length,embedding_size):
    print('start get_sen_vec')
    jieba_cut_list,content_list=get_content()
    word2vec_model = word2vec.Word2Vec(jieba_cut_list, min_count=1, window=2, size=50)
    all_doc_vec=[]
    for cut_content in jieba_cut_list:
        one_doc_vec = []
        for i,word in enumerate(cut_content):
            if i>=sentence_length:
                break
            one_doc_vec.append(list(word2vec_model[word]))
        sen_len=len(cut_content)
        if sen_len<sentence_length:
            cha=sentence_length-sen_len
            for i in range(cha):
                one_doc_vec.append(list(np.zeros((embedding_size,))))
        a = position_encoding(sentence_length, embedding_size)
        new_one_doc_vec=np.array(one_doc_vec)
        all_doc_vec.append(list(np.sum(new_one_doc_vec*a,axis=1)))
    print('final get_sen_vec')
    return np.array(all_doc_vec)
# if __name__ == '__main__':
#     get_content()
