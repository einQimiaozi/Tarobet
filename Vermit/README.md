# Vermit模块
    
  - 基于word2vec架构实现的词向量训练模型
    
  - ## 说明
    
   - 包含了skip gram和cbow两种模型，cbow的速度比skip gram快很多
     
   - 包含了通过单词求相似单词和通过向量求相似单词的两种相似度计算方法
     
   - 算法强制使用negative sample，但subsapmleing可以选择开启
     
 - ## api
   
   - Word2vec(word_dim:int,window=5:int,Corpus_path:str="",Corpus=[]:list,negative_sample_nums=5:int,table_size=10**8:int,learning_rate=0.025:float,epochs=5:int,k=10000:int,sample=1e-3:float)
     
     - 创建一个Word2vec模型并初始化，可以使用list作为语料输入，也可以直接读取文本文件，大多数超参数就是传统的word2vec的参数，k是sigmoid查表时划分的段落数
       
     - sample建议1e-6到1e-3之间，置为0时不开启subsampleing
     
     - 返回一个Word2vec对象，使用该对象才可以进行训练及相似度计算
       
   - Word2vec.train(mode="cbow":str)  开始训练，训练时会显示epochs和每个epochs的进度,mode值可以设置为"skip_gram"和"cbow"两种
     
   - Word2vec.similar(word:str,topn:int)  计算词典中和当前单词最相近的topn个词，如果当前单词不在词典中，则返回None
     
   - Word2vec.getSimilarWordForVector(vector:list,topn=10:int)  通过vector计算和当前vector相似的topn个单词
     
   - Word2vec["word":str] 返回word对应的词向量，若word不在字典中则返回None
     
   - "word":str in Word2vec  判断word是否存在于词典中
     
 - tips：求相似度时会输出一个分数，这个分数我是随便写的，之后会更新一个更好的度量方法，目前看看就好
   
 - ## 使用案例
   
   - 首先导入模块
   ```python
   from Vermit.vermit import Word2Vec
   ```
     
   - 初始化并训练
   ```python
   w2v = Word2Vec(Corpus_path='...',word_dim=400,window=5,epochs=5,sample=1e-5)
   w2v.train()
   ```
     
   - 进行相似度计算
   ```python
   w2v.similar("流感病毒")
   w2v.similar("临床应用")
   w2v.similar("气候")
   ```
     
   - 输出结果
   ```
   similar word is '流感病毒'
   ===========================================>
   [ 犬流感病毒 : 0.9999615770924172 ]
   [ 重组 : 0.9998809656252192 ]
   [ 以猪 : 0.9998381420676608 ]
   [ 甲型 : 0.9998069369063884 ]
   [ 同时 : 0.9996990306486604 ]
   [ 改善 : 0.999694833423572 ]
   [ 根据 : 0.999694500071264 ]
   [ 科学依据 : 0.9996913454968765 ]
   [ 问题提出 : 0.9996893951837295 ]
   [ 最为 : 0.9996874976174147 ]

   similar word '临床应用' not in vocab
   similar word is '气候'
   ===========================================>
   [ 气候指数 : 0.9995239700005324 ]
   [ 共存 : 0.9990845937040675 ]
   [ 体为 : 0.9990605676294161 ]
   [ 古盐度 : 0.9990353614434253 ]
   [ 湿润 : 0.9988319061019053 ]
   [ 安全性 : 0.9988036373236434 ]
   [ 水分 : 0.998801607560965 ]
   [ 作为 : 0.9987999999977679 ]
   [ 因子 : 0.9987967014488666 ]
   [ 持续 : 0.9987958439958227 ]
   ```
    
  - 顺便说一下，因为语料使用的是实验室的数据，不方便外泄，所以这里就不上传了
