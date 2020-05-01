# Poseg模块用于分词
  - ### 2020.05.01 内置ac自动机和dat树两种引擎，其中ac自动机偷懒了，只能拿来学习用，实际跑起来速度及慢
  - ## 使用方法
       ```python
       from Poseg import poseg
       ```
       导入Poseg模块
       ```python
       #dat树引擎
       word_seg = poseg.load_engine_DatTire()
       #ac自动机引擎
       word_seg = poseg.load_engine_Ac_automaton()
       ```
       选择需要的引擎加载
       ```python
       res = word_seg.segment(text)
       ```
       对制定文本text进行分词，返回一个list
       
  - ## 使用案例
       ```python
       from Poseg import poseg
       #dat树
       word_seg1 = poseg.load_engine_DatTire()
       text1 = '工商部联合工商行政管理局执法'
       text2 = '一万九千下岗工人再就业率提高'
       text3 = 'C++经常用于信息技术领域'
       text4 = '我不想开学，我不想隔离'
       res1 = word_seg1.segment(text1)
       res2 = word_seg1.segment(text2)
       res3 = word_seg1.segment(text3)
       res4 = word_seg1.segment(text4)
       print(res1)
       print(res2)
       print(res3)
       print(res4)
       #ac自动机
       word_seg2 = poseg.load_engine_Ac_automaton()
       res = word_seg2.segment('工商部联合工商行政管理局执法')
       print(res)
       ```
       
       ### 最终输出结果(结果依赖与内置字典)
       dat树引擎
       ```
       ['工商部', '联合', '工商行政管理局', '执法']
       ['一万九千', '下岗工人', '再就业率', '提高']
       ['C++', '经常', '用于', '信息技术', '领域']
       ['我', '不想', '开学', '，', '我', '不想', '隔离']
       ```
       ac自动机引擎
       ```
       Dictionary loading completed
       Init finish,the numbers of words in the Dictionary is 349046
       starting build.....
       building finish
       ['工商部', '联合工', '商行政管理局', '执法']
       ```
  - ## 关于词典
       - 词典保存在Dictionary文件夹下，文件名为Dictionary.txt
       - 内置词典可更换，更换后dat树引擎会自动更新双数组，所以更换后第一次运行需要一定时间加载
       - 词典使用jieba内置词典(感谢jieba开发者)，后续会试试看能不能找到其他更好的免费开源词典
  - ## 关于ac自动机引擎的分词错误
       - 由于ac自动机我偷懒了....所以分词算法使用了简单的正向前缀匹配，所以ac自动机不推荐大家使用，可以看看源码一起学习交流
       - 后续可能会更新，也可能会直接把ac自动机加入dat树中，不过ac自动机的代码会保留，只是不再作为分词工具的引擎
