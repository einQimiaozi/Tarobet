# Tarobet:一个基于python的自然语言处理工具箱
  - ## 包含以下工具
       - 分词工具Poseg
       - Transformer架构及MultiHeadAttention
       - 词向量编码(未上线)
       - 降维方法(未上线)
       - seq2seq模型(未上线)
       - 关键词提取方法(未上线)
  - ## 2020.08.10 update
       - 基于深度学习框架keras的Transformer架构包Foolmer上线，具体使用方法详见包内readme
  - ## 2020.05.01 update
       - 分词工具Poseg测试版上线，包含dat树和ac自动机两个分词引擎
       - ac自动机仅用作学习交流(偷懒没将数据结构写入闪存，每次需要重新构建字典树，速度及其慢)
       - dat树引擎可以直接使用，详细教程见Poseg的README.md
       - 其他模块暂时未开始开发，尽情期待！
       - 欢迎大家留言测试找bug
