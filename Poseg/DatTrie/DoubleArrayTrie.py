import os
import numpy as np
import hashlib

#DoubleArrayTrie，用于使用两个数组压缩字典书，降低空间复杂度加快查找速度
#使用numba加速,依赖numba模块
#根据numba官方说明，已经放弃支持python原生容器，下一个版本会将全部原生容器替换为njit版本

#2020.05.01     已添加对base和check数组创建算法的详细注释，包括启发式搜索，更新逆向匹配分词算法，使用jieba字典(免费开源字典真不好找啊.....)，支持替换内置字典使用自定义字典
#2020.04.23     dat树雏形已经建立，其中对状态值的搜索利用了komiya的启发式搜索算法
#dat树本质上是一个数组，并非一个真正的树形结构

#关于dat树的说明
# base表记录每一个Trie节点
# check用于查询当前base进行状态转移是否合法
# base[s] + c.code = t
# check[t] = base[s]
# s为当前状态的下标，t是要转移的状态的下标，c为当前字符的值(编码)
# 满足以上条件则可以进行状态转移
# base[s]的值需要由check[s+c.code] = 0来确定,当check[s+c.code] = 0时该元素没有被占用

#dat树的每个节点
class Node(object):
    #初始化节点，每个节点包含自己的子节点和自己的值value，每个节点对应一个字符
    #value可以根据需求存放不同的值,单纯分词任务可以存放bool类型，词性标注可以存放词性......
    def __init__(self,code,depth,lchild_pos,rchild_pos):
        self.code = code
        self.lchild_pos = lchild_pos
        self.rchild_pos = rchild_pos
        self.depth = depth

class datTrie(object):
    def __init__(self):
        self.Absolute_path = os.path.dirname(os.getcwd())+'/Poseg/Dictionary/'.strip()
        self.max_size = 2097152
        self.check = [0]*self.max_size
        self.base = [0]*self.max_size
        self.size = 0       #记录总共用的空间大小
        self.used = [False]*self.max_size
        self.base[0] = 1
        self.check[0] = 0
        self.corpus = []
        self.nextCheckPos = 0
        self.init()

    #对数组进行动态扩容
    def resize(self,size:int):
        if size <= self.size:
            raise ValueError("new size should be greater than current size.")
        base = [0]*size
        check = [0]*size
        used = [False]*size
        for i in range(len(self.base)):
            base[i] = self.base[i]
            check[i] = self.check[i]
            used[i] = self.used[i]
        del self.base
        del self.check
        del self.used
        self.base = base
        self.check = check
        self.used = used
        return True

    #搜索parent节点的子树们
    def fetch(self,parent:Node):
        siblings = []
        depth = parent.depth
        index = parent.lchild_pos
        #利用rchild指针的自加来记录当前节点的子节点个数(同一个字开头的单词个数)
        while index < parent.rchild_pos:
            word = self.corpus[index][depth:]
            if word == '':
                node = Node(code=-1,depth=depth+1,lchild_pos=index,rchild_pos=index+1)
                siblings.append(node)
            else:
                code = ord(word[0])
                #这里不能直接用parnet.siblings[-1].code == code来判断，因为若siblings为空，则siblings[-1]数组越界
                if len(siblings) == 0 or siblings[-1].code != code:
                    node = Node(code=code, depth=depth + 1, lchild_pos=index,
                                rchild_pos=index + 1)
                    siblings.append(node)
                else:
                    siblings[-1].rchild_pos += 1
            index+=1
        return siblings
    #这部分负责查找每个节点合适的base状态值，使用了启发式搜索，我承认这部分是从komiya大神的darts-java里抄的，我是真没看懂.....
    #这里如果不想使用启发是搜索，那么另begin从0开始的穷举法也是可以的，效率略低
    def insert(self, siblings, parent_base_index:int):
        begin = 0   #当前状态的base值
        first_empty = True   #first_empty是个开关，用于辅助记录第一个未被使用的t的位置(该t未必满足check[t] == 0)
        t = max(siblings[0].code+1, self.nextCheckPos) - 1  #初始化一个合适的t用于查找begin,nextcheckpos记录了当前这一群兄弟节点中第一个
                                                            #还没被占用的位置t，这么做的理由下面会说明
                                                            #加入max的原因没我太搞明白

        count = 0  #统计当前check数组中被使用的位置数量
        self.used[parent_base_index] = True
        find_begin = False
        #这部分主要是找到一个begin，检查check[t]的合法性，即check[t] == 0成立，且未发生数组越界等问题
        while not find_begin:
            t += 1
            if t >= self.max_size:
                raise Exception("value greater than max_size,please resize it")
            if self.check[t] != 0 or self.used[t]:
                count += 1
                continue
            elif first_empty:
                self.nextCheckPos = t  #标记第一个未被使用的位置
                first_empty = False
            begin = t - siblings[0].code    #这里实际上就是base[s]+code=t,反推出s，即begin
            if begin + siblings[-1].code >= self.max_size:
                raise Exception("value greater than max_size,please resize it")
            if self.used[begin]:
                continue
            if len(siblings) == 1:
                find_begin = True
                break
            #判断剩下的兄弟节点是否都满足check[begin+code] == 0
            for sibling in siblings[1:]:
                if self.check[begin + sibling.code] == 0 and self.used[begin + sibling.code] is False:
                    find_begin = True
                else:
                    find_begin = False
                    break
        #通过上面的算法找到了begin，即base[s]的值
        #下面处理一些细节

        #首先使用简单的启发式搜索算法，利用之前求出的nextcheckpos
        #由于nextcheckpos未被使用，所以在下一次查找t的时候直接从nextcheckpos开始检查其他兄弟节点是否符合check[t] == 0
        #但是如果当前已经找到的t和当前这一层第一个可以使用的t之间的位置已经大多数都被使用了
        #那么下一次从nextcheckpos开始查找符合条件的t时，在查找到当前这一层的t之前几乎都要一直执行t+=1，效率低下
        #所以可以设置一个比例，例如0.9,当当前的t和nextcheckpos之间被占用的位置大于等于两个位置之差的百分之90
        #则下次直接从t开始搜索，而不考虑nextcheckpos这个位置了
        if (count / (t - self.nextCheckPos + 1)) >= 0.9:
            self.nextCheckPos = t
        self.used[begin] = True
        #更新当前占用空间大小
        if self.size < begin + siblings[-1].code + 1:
            self.size = begin + siblings[-1].code + 1
        #设置check数组
        for sibling in siblings:
            self.check[begin + sibling.code] = begin
        #这部分就直接递归向下搜索插入了
        for sibling in siblings:
            if sibling.code == -1:
                self.base[begin + sibling.code] = -1 * sibling.lchild_pos - 1
            else:
                next_sibings = self.fetch(sibling)
                offset = self.insert(next_sibings, begin + sibling.code)
                self.base[begin + sibling.code] = offset
        return begin

    #更换内置字典将重新创建base和check数组，需要执行reload方法
    def reload(self):
        self.load(self.Absolute_path+'Dictionary.txt')
        self.corpus = sorted(list(set(self.corpus)))  # 对字典去重排序
        print('Dictionary loading completed')
        self.root = Node(None, depth=0, lchild_pos=0, rchild_pos=len(self.corpus))
        # 开始查找当前节点的子节点们并据算base[s]的值
        siblings = self.fetch(self.root)
        self.insert(siblings, 0)
        print('Init finish,the numbers of words in the Dictionary is ' + str(len(self.corpus)))
        del self.corpus  # 释放内存
        base = np.array(self.base)
        np.save(self.Absolute_path+'base.npy', base)
        check = np.array(self.check)
        np.save(self.Absolute_path+'check.npy', check)

    #构建dat树
    def init(self):
        #如果字典的md5变动或首次加载，则创建base和check数组
        if os.path.exists(self.Absolute_path+'base.npy') and os.path.exists(self.Absolute_path+'check.npy'):
            with open(self.Absolute_path+'Dictionary.txt', 'rb') as f:
                md5_base = hashlib.md5(f.read()).hexdigest()
            with open(self.Absolute_path+'MD5/Dictionary_md5', 'r', encoding='utf-8') as f:
                md5_base_check = f.read().replace('\n','')
            if md5_base != md5_base_check:
                self.reload()
                with open(self.Absolute_path+'MD5/Dictionary_md5', 'w', encoding='utf-8') as f:
                    f.write(md5_base)
            else:
                self.base = np.load(self.Absolute_path+'base.npy').tolist()
                self.check = np.load(self.Absolute_path+'check.npy').tolist()
        else:
            self.reload()

    #查找输入是否为一个单词
    def isWord(self, word:str):
        pos = 0  #root指针
        if word == '':
            return False
        for c in word:
            c = ord(c)
            #状态转移到下一个状态
            next = abs(self.base[pos]) + c
            #判断当前状态是否越界
            if next > self.max_size:
                return False
            #判断状态转移的合法性
            if self.check[next] != abs(self.base[pos]):
                return False
            pos = next  #执行状态转移
        #判断当前base值是否到到达词尾，并且状态转移合法
        if self.base[self.base[pos] - 1] < 0 and self.base[pos] == self.check[self.base[pos] - 1]:
            return True
        else:
            return False

    #分词方法，使用逆向匹配
    def segment(self,text:str):
        if not isinstance(text,str):
            raise ValueError('Parameter \'text\' should be a str type,not '+str(type(text)))
        res = []
        text_size = len(text)-1
        r = text_size
        while r>=0:
            max_word = text[r]
            for l in range(0,r):
                word = text[l:r+1]
                if self.isWord(word):
                    if len(word)>len(max_word):
                        max_word = word
                        break
            res.insert(0,max_word)
            r = r-len(max_word)
        return res

    #加载预定义字典
    def load(self,corpus:str):
        if not isinstance(corpus,str):
            raise ValueError('Parameter \'corpus\' is Dictionary path ,it should be a str type,not '+str(type(corpus)))
        if corpus[-4:] !='.txt':
            raise ValueError('Parameter \'corpus\' should end with \'.txt\'')
        with open(corpus,'r',encoding='utf-8') as f:
            word_base = f.readlines()
        for word in word_base:
            word = word.split(' ')[0].replace('\n','')
            self.corpus.append(word)
        return True


                    
