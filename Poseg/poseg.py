from Poseg.DatTrie.DoubleArrayTrie import datTrie
from Poseg.AC_automaton.AcAutomaton import Trie

def load_engine_DatTire():
    dat = datTrie()
    return dat
def load_engine_Ac_automaton():
    ac_auto = Trie()
    ac_auto.load()
    ac_auto.build()
    ac_auto.AhoCorasick()
    return ac_auto

