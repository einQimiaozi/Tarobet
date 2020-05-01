# Poseg模块用于分词
  - ## 2020.05.01 内置ac自动机和dat树两种引擎，其中ac自动机偷懒了，只能拿来学习用，实际跑起来速度及慢
  - ## 使用方法
       ```python
       from Poseg import poseg
       ```
       导入Poseg模块
       ```python
       #dat树引擎
       word_seg1 = poseg.load_engine_DatTire()
       #ac自动机引擎
       word_seg2 = poseg.load_engine_Ac_automaton()
       ```
       选择需要的引擎加载
