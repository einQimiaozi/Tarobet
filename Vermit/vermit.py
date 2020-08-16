import numpy as np
from random import randint,random
from collections import Counter
from scipy.spatial.distance import cosine


class Word2Vec(object):
    def __init__(self,word_dim,window=5,Corpus_path:str="",Corpus=[],negative_sample_nums=5,table_size=10**8,learning_rate=0.025,epochs=5,k=10000,sample=1e-3):
        assert Corpus_path or Corpus , "The parameter is wrong, 'Corpus_path' and 'Corpus' must be passed in at least one"
        if not Corpus_path:
            self.corpus = Corpus
        else:
            self.corpus = self.load(Corpus_path)
        self.count_corpus = Counter(self.corpus)
        del self.count_corpus['</s>']
        self.window = window
        self.word_dim = word_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        #建立词序号字典
        self.vocab = {word:index for index,word in enumerate(self.count_corpus.keys())}
        if sample>0:
            self.sample = sample
            self.sub_sampling()
        print("total of {} words in corpus".format(len(self.vocab)))
        # 负采样辅助向量长度
        self.table_size = table_size
        # 计算辅助向量
        self.array = self.init_array(shuffle=True)
        # 否定词数量
        self.negative_sample_nums = negative_sample_nums
        # 初始化词向量
        self.word_vec = self.init_wordvec()
        self.theta = self.init_theta()
        # 自适应学习率超参数
        self.minlr = 1e-4
        self.proecssed_num = 10000
        # 建立sigmoid表,sigmoid不用计算，提前建好表即可
        self.sigmoid_bulid(k=k)
    def load(self,path):
        # 原语料库大小
        self.Corpus_MAX_LENGTH = 0
        # 换行符数量
        self.breaks_num = 0
        res = []
        with open(path,'r',encoding='utf-8') as f:
            for sencentes in f.readlines():
                sencentes.replace('\n',"").replace("\r\n","")
                for word in sencentes.split(' '):
                    self.Corpus_MAX_LENGTH+=1
                    res.append(word.strip('\n').strip('\r\n'))
                res.append("</s>")  # 换行符
                self.breaks_num+=1
        return res
    # 随机初始化每个词作为中心词的向量(代替one-hot),res=(random/random_max-0.5)/word_dim
    def init_wordvec(self):
        return (np.random.rand(len(self.count_corpus),self.word_dim)-0.5)/self.word_dim
    # 初始化theta的权重
    def init_theta(self):
        return np.zeros(shape=(len(self.count_corpus),self.word_dim))
    # 初始化辅助向量,以词频作为依据,词频越高在负采样时被选中的概率越大
    def init_array(self,shuffle=False):
        start = 0
        array = [0 for _ in range(self.table_size)]
        for word in self.vocab.keys():
            # counter(w)**(3/4)counter(u)**(3/4),u为所有词频相加(即未去重的原语料库大小)
            end = round((self.count_corpus[word]**(0.75))/(self.Corpus_MAX_LENGTH**(0.75)))*self.table_size
            array[start:end] = [self.vocab[word] for _ in range(end-start)]
            start+=end
        # 打乱
        if shuffle:
            np.random.shuffle(array)
        return array
    # 获取窗口上下文，使用随机窗口
    def get_context_index(self,word_index,maxlen):
        magic_window = randint(1,self.window)
        left = max(0,word_index - magic_window)
        right = min(maxlen,word_index + magic_window)
        return left,right
    # 负采样，返回的是index不是词
    def negative_sample(self,current_word):
        neg_index = []
        while len(neg_index)<self.negative_sample_nums:
            index = randint(0,self.table_size-1)
            if index == self.array[self.vocab[current_word]]:
                continue
            neg_index.append(self.array[index])
        return neg_index
    # 自适应学习率
    def Adaptive_learning(self,processed_index):
        self.learning_rate = max(self.minlr,self.learning_rate*(1-processed_index/(self.Corpus_MAX_LENGTH+1)))
    # 查表加速
    def sigmoid_bulid(self,k=10000):
        self.k = k
        start = -6
        step = 12/k
        self.sigmoid_table = []
        for _ in range(k):
            self.sigmoid_table.append(1 / (1+np.exp(-start)))
            start+=step

    def sigmoid(self,x):
        if x <= -6:
            return 0
        elif x >= 6:
            return 1
        else:
            sig_index = int(self.k*((x+6)/12))
        return self.sigmoid_table[sig_index]
    # 高频词降频
    def sub_sampling(self):
        # 降高频后的新词典
        new_vocab = dict()
        index = 0
        for word in self.vocab.keys():
            word_length = self.count_corpus[word]/self.Corpus_MAX_LENGTH
            # 被丢弃的概率
            prob = (self.sample/word_length)**0.5+(self.sample/word_length)
            if random() <= prob:
                new_vocab[word] = index
                index+=1
            else:
                while word in self.corpus:
                    self.corpus.remove(word)
                    self.Corpus_MAX_LENGTH -= 1
        del self.vocab
        self.vocab = new_vocab
    # skip_gram
    def TrainForSkipGram(self):
        ss = 0  # 句子开头
        se = 0  # 句子末尾
        while se<len(self.corpus):
            if self.corpus[se] != "</s>":
                se+=1
            else:
                sencentence = self.corpus[ss:se]
                for pos,word in enumerate(sencentence):
                    i = pos+ss
                    if i%1000==0 or i==self.Corpus_MAX_LENGTH-1:
                        print('\r', round((i+1)/self.Corpus_MAX_LENGTH*100)*"♥", end=' ==>  {}%'.format(round((i+1)/self.Corpus_MAX_LENGTH*100)))
                    if i%self.proecssed_num==0:
                        self.Adaptive_learning(i)
                    # 获取上下文(包括中心词)
                    context_start,context_end = self.get_context_index(pos,se-ss)
                    for c in range(context_start,context_end):
                        # 当前上下文的第c个单词，skip_gram是一个一个词处理的
                        c_word = sencentence[c]
                        # 当前上下文在向量中的位置
                        context_word = self.vocab[c_word]
                        # 梯度缓存器
                        e = 0
                        # 负采样,和cbow不同，我们要进行len(context(w))次负采样，可以理解为对context中每个词进行一次负采样，但是由于最终我们希望通过中心词得出上下文，所以对中心次做多次即可
                        neg_sample = self.negative_sample(word)
                        # 组合成train数据，位置0为正样本，其他位置为负样本
                        train_data = [self.vocab[word]] + neg_sample
                        for sample in range(len(train_data)):
                            # 本次要更新的权重
                            current_theta_index = train_data[sample]
                            # sigmoid
                            sig = self.sigmoid(np.dot(self.word_vec[context_word],self.theta[current_theta_index].T))
                            # 计算梯度
                            # 正例
                            if sample == 0:
                                gradient = (1 - sig)*self.learning_rate
                            # 负例
                            else:
                                gradient = -sig*self.learning_rate
                            # 梯度叠加
                            e += gradient * self.theta[current_theta_index]
                            # 更新权重矩阵
                            self.theta[train_data[sample]] += gradient * self.word_vec[context_word]
                        # 更新上下文词向量
                        self.word_vec[context_word] += e
                # 处理句子位置
                ss = se+1
                se = ss

    def train(self,mode="skip_gram"):
        if mode == "skip_gram":
            for i in range(self.epochs):
                print("\nepochs {}".format(i+1))
                self.TrainForSkipGram()
        elif mode == "cbow":
            assert False, "Sorry, the cbow model hasn't been written yet..."
        else:
            assert False,"The parameter 'mode' is wrong and must be a string type with 'skip_gram' value of a or 'cbow'."
        print("\n")
    # 计算单词相似度
    def similar(self,word,topn=10):
        if word not in self.vocab:
            print("similar word '{}' not in vocab".format(word))
            return None
        print("similar word is '{}'".format(word))
        # 获取待计算词向量
        index = self.vocab[word]
        wordvec = self.theta[index]
        words = dict()
        distence = []
        res = {}
        for word,i in self.vocab.items():
            if i==index:
                distence.append(1)
                words[1] = word
            else:
                temp = cosine(self.theta[i],wordvec)
                distence.append(temp)
                words[temp] = word
        distence.sort()
        for i in range(topn):
            res[words[distence[i]]] = (1-distence[i])**self.epochs
        print("===========================================>")
        for word,dis in res.items():
            print("[ {} : {} ]".format(word,dis))
        print('\n')
        return res
