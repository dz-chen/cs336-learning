import os
import heapq
import regex as re
import json
from tests.common import gpt2_bytes_to_unicode


########################### 辅助类 ###########################
class _Node:
    def __init__(self, val:bytes, pretoken:bytes):
        self.val = val              # node对应的token值
        self.pretoken = pretoken    # 当前node所属pretoken!!!
        self.prev = None
        self.next = None

class _DoubleLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add_first(self, val):
        pass

    def add_last(self, node:_Node):
        if self.tail is None:
            self.head = node
            self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

    # 正向遍历
    def print_forward(self):
        res = []
        cur = self.head
        while cur:
            res.append(str(cur.val))
            cur = cur.next
        print(" <-> ".join(res))


class _HeapPair:
    """记录pair大小的大根堆使用"""
    def __init__(self, pair:tuple[bytes,bytes], cnt:int):
        self.pair = pair
        self.cnt = cnt

    # 自定义比较规则(heap默认小根堆, 需要大根堆这里改写比较规则)
    def __lt__(self, other):
        # 大根堆：self.cnt > other.cnt  若相等则按字典序比较 pair
        return (self.cnt, self.pair) > (other.cnt, other.pair)

    def __repr__(self):
        return f"_HeapPair(pair={self.pair}, cnt={self.cnt})"



########################### tokenizer ###########################
class Tokenizer():
    def train():
        pass

    def save():
        pass
    
    def load():
        pass

    def encode():
        pass

    def decode():
        pass

class BPETokenizer(Tokenizer):

    def __init__(self, name):
        super().__init__()     
        # 核心 
        self.vocab:dict[int,bytes] = {}         # id->bytes
        self.vocab_inv:dict[bytes,int] = {}     # bytes->id
        self.merges:list[tuple[bytes, bytes]] = []
        self.special_tokens:list[str] = []

        # 训练时用到
        self.pretoken_cnt:dict[bytes, int] = {}  # eg: newest:2 . pretoken 通常是一个单词, 统一用字节表示
        self.pretoken_tokenlist:dict[bytes,_DoubleLinkedList] = {}  # eg: newst: n<=> e <=> w <=> e <=> s <=> t
        self.token_loclist:dict[bytes,list[_Node]] = {}             # 获取 token 在 pretoken_tokenlist 链表中的具体节点. 方便合并时对前后进行更新(注意一个token可能出现在多个链表中)
        self.pair_cnt:dict[tuple[bytes,bytes],int] = {}             # eg:    (e,st) -> 2
        self.pair_heap = []                     # 延迟删除的大根堆. 元素是否还有效需与pair_cnt 对比看是否相等

    def _init_vocab(self):
        for i in range(0,256):
            self.vocab[i] = bytes([i])
        for s in self.special_tokens:
            self.vocab[len(self.vocab)] = bytes(s,"utf8")

    def _split_sentence(self,doc_path:str):
        """ 
        TODO: 输入文件非常大时的优化
        """
        text = None
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            raise Exception("read file %s failed. err:%s "%(doc_path, e))

        # 一个 sentence 指一个完整的文本
        sentences = re.split(text, "|".join(self.special_tokens))
        return sentences
    

    def _pre_tokenize(self, sentence):
        """
        将一个已分割special token 后的句子, 预分割为独立单元(比如单词,数字,标点. 这里统称 pre_tokens)
        同时统计 pretoken 相关的初始信息
        TODO:
            注意空格也会成为token或者token的一部分
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # 遍历所有pretoken(单词), 获取全局的 pretoken_cnt,pretoken_tokenlist,token_loclist,pair_cnt,pair_heap
        # TODO: 这里的  pretoken, pair 统计有误,待纠正 TODO 
        for it in re.finditer(PAT, sentence):
            pretoken = bytes(it.group(),"utf8")

            self.pretoken_cnt[pretoken] = self.pretoken_cnt[pretoken] + 1
            
            dll = _DoubleLinkedList()
            n = len(pretoken)
            for i,b in enumerate(pretoken):
                byte = bytes([b])
                node = _Node(byte, pretoken)
                dll.add_last(node)
                self.token_loclist[byte].append(node)   # 含有token的所有node(比如 a 可能出现在 [node1, node2,node3...])
                if i != n-1:
                    pair = (byte, bytes(pretoken[i+1]))
                    self.pair_cnt[pair] = self.pair_cnt[pair] + 1
                    
            self.pretoken_tokenlist[pretoken] = dll

        # pair_heap 没法在遍历 pretoken时更新, 最终统一遍历一次 pair_cnt 进行初始化
        for pair,cnt in self.pair_cnt.items():
            heapq.heappush(self.pair_heap, _HeapPair(pair,cnt))

    def _merge(self,vocab_size:int):
        niter = vocab_size - 255 - len(self.special_tokens)
        for i in range(niter):
            # 对于所有 X a b Y, pair为(a,b)的情形. 合并 ab为c, 更新链表为 X c Y
            heap_pair = heapq.heappop(self.pair_heap)   # 次数最大的pair
            pair, cnt = heap_pair.pair, heap_pair.cnt   
            while self.pair_cnt[pair] != cnt:           # 不等说明堆顶pair已失效
                heap_pair = heapq.heappop(self.pair_heap)
                pair, cnt = heap_pair.pair, heap_pair.cnt
                new_token = pair[0] + pair[1]           # (a,b) 合并为新 token ab

                self.merges.append((pair[0],pair[1]))
                self.vocab[len(self.vocab)] = new_token

                newpairs_cnt = {}
                for node_a in  self.token_loclist[pair[0]]: # 更新pair所在链表 && 新pair计数
                    """
                    这里每一个遍历,近似等于对一个 pretoken 中 ab 的处理
                    TODO: 是否存在 X a b a b Y , 且需要合并 ab 的情况,该如何处理!!!! 
                    """
                    node_b = node_a.next
                    if node_b is None or node_b.val != pair[1]:
                        continue
                    node_X = node_a.prev
                    node_Y = node_b.next
                    pretoken = node_a.pretoken
                    node_new = _Node(new_token, pretoken)
                    self.token_loclist[new_token].append(node_new)
                    # TODO: 头节点/尾节点可能还需要特殊处理,去掉引用
                    if node_X is not None:
                        node_new.prev = node_X
                        node_X.next = node_new
                        pair_Xab = (node_X.val, new_token)
                        self.pair_cnt[pair_Xab] = self.pair_cnt[pair_Xab] + self.pretoken_cnt[pretoken]
                        newpairs_cnt[pair_Xab] = self.pair_cnt[pair_Xab]
                    if node_Y is not None:
                        node_new.next = node_Y
                        node_Y.prev = node_new
                        pair_abY = (new_token, node_Y.val)
                        self.pair_cnt[pair_abY] = self.pair_cnt[pair_abY] + self.pretoken_cnt[pretoken]
                        newpairs_cnt[pair_abY] = self.pair_cnt[pair_abY]
                    self.token_loclist[new_token].append(node_new)

                    # 更新pairheap
                    for pair, cnt in newpairs_cnt.items():
                        heapq.heappush(self.pair_heap, _HeapPair(pair,cnt))
        
    def train(self,input_path:str, 
                vocab_size:int, 
                special_tokens: list[str]
            ) -> tuple[dict[int, bytes],  list[tuple[bytes, bytes]]]:
        """
        xxxxxxxx return
            vocab, merges
            
        xxxxxxxx vocab_size
            vocab_size = 初始字节词表(256) + 特殊token数量 + merge产生的token数(每次merge产生一个)
        """
        self.special_tokens = special_tokens
        self._init_vocab()
        sentences = self._split_sentence(input_path)
        for sentence in sentences:
            self._pre_tokenize(sentence)
        self._merge(vocab_size)
        return self.vocab, self.merges

    def save(self, vocab_path, merges_path, special_tokens_path):
        # vocab (unicode). 注意存储的是 str -> id, 与 self.vocab 顺序不同
        # gpt2_byte_encoder = {k: v for k, v in gpt2_bytes_to_unicode().items()}
        # TODO: 待检查是否有问题
        vocab_inv = {token.decode("utf-8"): idx 
                        for idx, token in self.vocab.items() }  # bytes->id
        with open(vocab_path, "w", "utf-8") as vocab_f:
            json.dump(vocab_inv, vocab_f)


        # merges (unicode)
        gpt2_bpe_merges = []
        with open(merges_path, "w", "utf-8") as merges_f:
            for merge in self.merges:
                merges_f.write("%s %s\n"%(merge[0].decode("utf-8"), merge[1].decode("utf-8")))

        # special_rokens (str)
        with open(special_tokens_path, "w", "utf-8") as sptk_f:
            for sptk in self.special_tokens:
                merges_f.write("%s\n"%(sptk))


    def load(self, vocab_path, merges_path, special_tokens_path):
        """
        参考自 tests/common.py
        """
        # vocab (byte-encoded)
        with open(vocab_path) as vocab_f:
            self.vocab_inv = json.load(vocab_f)     # bytes->id
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        self.vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in self.vocab_inv.items()
        }

        # merges (byte-encoded)
        gpt2_bpe_merges = []
        with open(merges_path) as merges_f:
            for line in merges_f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        self.merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]

        # special_rokens (str)
        with open(special_tokens_path) as sptk_f:
            for line in sptk_f:
                cleaned_line = line.rstrip()
                self.special_tokens.append(cleaned_line)


    def encode(self, text:str) -> list[int]:
        encoded_tokens = []
        # pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for it in re.finditer(PAT, text):
            pretoken = bytes(it.group(),"utf8") 
            tokens = [bytes([b]) for b in pretoken]
            
            # 对于每个merge,判断这个pretoken的当前子串是否能合并 (注意 merge 和 tokens 的内外循环关系)
            for (a, b) in self.merges:
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b: # 可以合并
                        new_tokens.append(a + b)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens
            # tokens 转换为id形式   TODO:vocab 中不存在的情形
            encoded_tokens.extend([self.vocab_inv[token] for token in tokens])
        return encoded_tokens

        
    def decode(self, ids:list[int]) -> str:
        UNDEF = "\uFFFD"
        decoded_tokens = []
        for id in ids:
            decoded_tokens.append(self.vocab[id] if id in self.vocab.keys() else UNDEF)
        text = "".join(decoded_tokens)
        return text

if __name__ == "__main__":
    print("to impl")