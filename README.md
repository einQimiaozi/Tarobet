# Tarobet:python自然语言处理工具箱
  - ## 包含以下工具
       - 分词工具--->Poseg
       - Transformer架构及MultiHeadAttention--->Foolmer
       - Word2vec模型--->Vermit
       - 关键词提取工具库--->Lovextras
       - seq2seq模型(未上线)
       - 生成式模型(未上线)
  - ## 2020.08.17 update
       - Word2vec模型上线，直接用，跟tensorflow没关系，优化方法直接使用negative sample
       - 新增两种关键词提取方法，一种基于tf idf算法，另一种使用word2vec模型训练词向量后查找聚类中心点
  - ## 2020.08.10 update
       - 基于深度学习框架keras的Transformer架构包Foolmer上线，具体使用方法详见包内readme
  - ## 2020.05.01 update
       - 分词工具Poseg测试版上线，包含dat树和ac自动机两个分词引擎

