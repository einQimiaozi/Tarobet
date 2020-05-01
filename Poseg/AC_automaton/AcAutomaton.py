import os
#自带ac自动机方法的前缀树
#使用ac自动机前必须要先构建好一颗前缀树，构建好后直接调用AhoCorasick方法即可对该树插入fail表
#每次插入ac自动机为完全重新插入，所以建议将前缀树完全构建好之后，再插入ac自动机
#该部分代码参考何晗老师的<自然语言处理入门>一书，在此声明并表示感谢

class Node(object):
    def __init__(self,value):
        self.value = value
        self.children = {}
        self.fail = None

    def add_child(self,char,value,overWrite=False):
        child = self.children.get(char)
        if child is None:
            child = Node(value)
            self.children[char] = child
        elif overWrite:
            child.value = value
        return child

class Trie(object):
    def __init__(self):
        self.root = Node(None)
        self.corpus = []
        self.Absolute_path = os.path.dirname(os.getcwd()) + '/Poseg/Dictionary/'.strip()

    def __setitem__(self, key, value):
        state = self.root
        for i,char in enumerate(key):
            if i<len(key)-1:
                state = state.add_child(char,None,False)
            else:
                state = state.add_child(char,value,True)

    def __getitem__(self, item):
        state = self.root
        for c in item:
            state = state.children.get(c)
            if state is None:
                return None
        return state.value
    #加载字典
    def load(self,corpus = None):
        if corpus and not isinstance(corpus,str):
            raise ValueError('Parameter \'corpus\' is Dictionary path ,it should be a str type,not '+str(type(corpus)))
        else:
            corpus = self.Absolute_path+'Dictionary.txt'
        if corpus[-4:] !='.txt':
            raise ValueError('Parameter \'corpus\' should end with \'.txt\'')
        with open(corpus,'r',encoding='utf-8') as f:
            word_base = f.readlines()
        for word in word_base:
            word = word.split(' ')[0].replace('\n','')
            self.corpus.append(word)
        print('Dictionary loading completed')
        print('Init finish,the numbers of words in the Dictionary is ' + str(len(self.corpus)))
        return True
    def build(self):
        print('starting build.....')
        for word in self.corpus:
            self[word] = word[0]
        del self.corpus
        print('building finish')

    def AhoCorasick(self):
        #使用队列对前缀树进行BFS
        queue = [self.root]
        while len(queue)>0:
            current_node = queue.pop(0)
            for char,child in current_node.children.items():
                queue.append(child)
                # fail=父节点->fail->相同前缀的子节点 or 根节点
                if current_node == self.root:
                    child.fail = self.root
                else:
                    p_fail = current_node.fail
                    while p_fail:
                        if char in p_fail.children:
                            child.fail = p_fail.children[char]
                            break
                        else:
                            p_fail = p_fail.fail
                    if not p_fail:
                        child.fail = self.root
        return True
    def segment(self,text:str):
        if not isinstance(text,str):
            raise ValueError('Parameter \'text\' should be a str type, not '+str(type(text)))
        res = []
        current_word = ''
        p = self.root
        i = 0
        while i < len(text):
            if text[i] in p.children:
                current_word+=text[i]
                p = p.children[text[i]]
            else:
                if len(current_word) > 0:
                    res.append(current_word)
                current_word = ''
                p = p.fail
                i-=1
            i+=1
        if len(current_word)>0:
            res.append(current_word)
        return res
