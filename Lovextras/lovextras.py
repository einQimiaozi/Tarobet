from math import log
import numpy as np
from scipy.spatial.distance import cosine

class Tf_Idf(object):
    # 初始化语料库请保证传入二维数据，维度1为文章，维度2为每篇文章中的单词
    def __init__(self,Corpus_path="",Corpus=[]):
        if Corpus_path:
            self.corpus = self.load(Corpus_path)
        elif Corpus:
            self.corpus = Corpus
        self.idf_vocab = self.idf()
        self.tf_vocab = []
        for doc in self.corpus:
            self.tf_vocab.append(self.tf(doc))
    def load(self,path):
        corpus = []
        with open(path,'r',encoding="utf-8") as f:
            for doc in f.readlines():
                corpus.append(doc.split(" "))
        return corpus
    def tf(self,doc):
        tf_vocab = dict()
        for word in doc:
            if word in tf_vocab:
                tf_vocab[word] += 1
            else:
                tf_vocab[word] = 1
        doc_length = len(tf_vocab)
        for word,freq in tf_vocab.items():
            tf_vocab[word] = freq/doc_length
        return tf_vocab
    def idf(self):
        idf_vocab = dict()
        for doc in self.corpus:
            for word in doc:
                idf_vocab[word] = 0
        doc_nums = len(doc)
        for word in idf_vocab.keys():
            for doc in self.corpus:
                if word in doc:
                    idf_vocab[word] += 1
            idf_vocab[word] = log(doc_nums/(idf_vocab[word]+1))
        return idf_vocab
    def train(self,topn=1):
        keywords = []
        for i,doc in enumerate(self.corpus):
            words = dict()
            for word in doc:
                tf = self.tf_vocab[i][word]
                idf = self.idf_vocab[word]
                words[word] = tf*idf*100
            # 按tf idf值排序
            keywords.append(sorted(words.items(), key=lambda d: d[1], reverse=True)[:topn])
        return keywords

class ExtractionForWordVector(object):
    def __init__(self,vectors,topn=1,mode="cosine"):
        self.vectors = vectors
        self.topn = topn
        assert mode == "euclidean" or mode == "cosine","The parameter 'mode' is wrong and the value of 'mode' must be euclidean or 'cosine'."
        self.mode = mode
    def Extraction(self,vectors):
        center = np.array(np.sum(vectors, axis=0))/len(vectors)
        dis = dict()
        # 这里保留i是为了保留每个向量在原文本中的位置，否则计算排序后向量和词无法一一对应
        if self.mode == "cosine":
            for i,vector in enumerate(vectors):
                dis[i] = cosine(vector, center)
        elif self.mode == "euclidean":
            for i,vector in enumerate(vectors):
                dis[i] = np.sqrt(np.sum((vector-center)**2))
        return sorted(dis.items(), key=lambda d: d[1], reverse=False)[:self.topn]
    def fit(self):
        res = []
        for vector in self.vectors:
            res.append(self.Extraction(vector))
        return res
