""" 1.张量运算基础回顾
xxxxxxxxxxxxxxxxxxx 概念
数字 --> 标量
一维 --> 向量
二维 --> 矩阵
多维 --> 张量


xxxxxxxxxxxxxxxxxxx 内积和外积(向量为例)
内积: 长度相同，同下标相乘,然后求和收缩. 结果是标量 (n,n-> )
外积: 长度不同,不同下标广播相乘, 不求和收缩. 结果是矩阵(m,n->m n)
物理意义: 内积（点积）描述「两个向量有多同向、投影有多大」，结果是标量;
外积(张量积 /outer product)描述「两个方向组合成一个平面 / 矩阵变换」,结果是张量,记录两两分量的耦合

xxxxxxxxxxxxxxxxxxx 张量外积
维度: 升维, 最终结果 p+q维
计算: 把两个张量的所有维度合并在一起,形成一个张量.计算所有维度的分量广播相乘
举例: 三维的 A(i,j,k) 三维的 B(l,m,n), 结果 C是六维的(i,j,k,l,m,n)
     其中 Cijklmn = Aijk*Blmn


xxxxxxxxxxxxxxxxxxx 张量缩并
维度: 最终结果 p+q-2
计算: 在一个张量或者两个张量的乘积(外积)里, 取两个长度相等的维度,对相同下标, 求和
前提要求: 被缩并配对的两个维度,长度必须相等!!!
典型例子: 矩阵乘法就是张量缩并
特别说明: 矩阵乘法是张量缩并,不是内积!! 但可以看成由多个向量内积组成. Cij = Aik*Bkj对k求和,
        这个Cij子元素的过程就是两个向量的内积(外积+收缩)
        而对于整个矩阵乘法来说,两个张量，共享下标 k,对 k 求和 = 缩并 => 属于缩并!!!
        (感觉这里可能没太理清楚 TODO)

xxxxxxxxxxxxxxxxxxx 张量内积
维度: p+q-2
本质是: 两个张量外积之后,恰好缩并两个维度,只是这两个维度相邻且长度相等(内积 = 外积 + 一次缩并)
补充说明: 内积缩并哪一对下标是 人为约定
        力学里默认是相邻下标; 深度学习里用 `einsum` 其实可以指定任意一对下标缩并
        比如 `ikj,klm->ijlm` 就是缩并第一个张量的第 2 个下标
"""

""" 2.PyTorch/einops 张量基础回顾
xxxxxxxxxxxxxxxxxxx 多维时在内存中的排列顺序
Tensor默认行优先(与Numpy一致),即:
- 逻辑 shape(看到的维度顺序): `(dim0, dim1, dim2,...)`
- 内存布局: C 连续时, 最右边的下标先跑满/或者说最后一维变化最快(如 a12,a13,a14,a15...)

xxxxxxxxxxxxxxxxxxx PyTorch 中权重(weight) 输入 x 的形状
- 权重 weight shape: [d_out, d_in]
- 偏置 bias shape: [d_out]
- 输入 x: 默认是 [..., d_in](最后一维是输入维度,即x是行向量,方便pytorch行优先的默认内存布局)
- 输出 y: [..., d_out]
- 计算的是 y = xWt (W的转置)
- 问题: 为何W的shape不定义为 [d_in, d_out], 直接 y=xW 省去人为转置? => 理论可以,但是pytorch有
        复杂的背景作出这个定义的..... 
"""

""" 3.网络基础概念回顾(归一化,正则化,softmax,激活函数,FFN.....)
xxxxxxxxxxxxxxxxxxx 归一化 Normalization
调整张量内部数值分布, 让训练更稳; 作用在特征上

xxxxxxxxxxxxxxxxxxx 正则化 Regularization
约束模型、抑制过拟合,防止记住训练集; 是对参数/模型的约束

xxxxxxxxxxxxxxxxxxx softmax
转成总和=1的概率分布, 一般用来算分类概率

xxxxxxxxxxxxxxxxxxx 激活函数
激活函数: 给网络 [引入非线性] 没有它,再多线性层也等价单层线性,只能学直线
线性单元(xxLU): 属于激活函数的一个分支,大多形式是分段、带截断的

xxxxxxxxxxxxxxxxxxx FFN
1.泛指信息单向流动,没有循环/反馈连接的神经网络; 和 RNN(循环,有反馈)相对
2.在LLM语境下,通常指 Transformer中的  输入 x(d_model) --> Linear1(4d) --> ReLU --> Linear2(d) --> 输出 
    这个结构及其变体

xxxxxxxxxxxxxxxxxxx Embedding 与  Encoding
Embedding = 可学习、查表得到的位置向量(参数)
Encoding = 公式计算出来的固定位置向量(无参数,或者说是超参数)
但日常交流里很多人不严格区分，统一叫 PE(position encoding/embedding)

"""



import os
import torch
import torch.nn as nn
from einops import einsum,rearrange
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, d_in, d_out, device=None, dtype=None):
        """
        d_in: int final dimension of the input
        d_out: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        # 按作业要求,不需要 bias;  初始化的分布根据讲义要求计算
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        mean, std = 0, torch.sqrt(2/(d_in + d_out))
        init.trunc_normal_(self.weight, mean, std, -3*std, 3*std)

        # TODO: device, dtype 设置
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            x:(...,d_in)
        """
        # y = xWt
        # 输出形状由 -> 右侧决定, 不由参数顺序决定. 以下两个写法是等价的!!!
        # 用 einsum 不需要人为对W取转置换, 自动感知
        # return einsum(self.weight, x , "d_out d_in, ... d_in -> ... d_out")
        return einsum(x, self.weight ,  "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: int Size of the vocabulary => 即 vocab_size
        embedding_dim: int Dimension of the embedding vectors, i.e., dmodel => 即 d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        # (vocab_size, d_model)
        self.embeddings = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        mean, std = 0, 1
        init.trunc_normal_(self.weight, mean, std, -3*std, 3*std)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        对每条输入, seq中的每个tokenid 在 embeddings中查表. 即 每个tokenid 对应embedding 的一行向量(embedding)
        输入 token_ids: (batch_size, seq_len)
        输出: (batch_size, seq_len, d_model)
        """
        # 也可以one-hot然后矩阵乘ebbeddings, 但内存浪费过大!!!
        return self.embeddings[token_ids]

class RMSNorm(nn.Module):
    """
    均方根层归一化
    """
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = nn.Parameter(torch.empty(self.d_model,))
        

    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        输出: (batch_size, seq_len, d_model)
        """

        # 提前转换为 float32 防止 square 时溢出
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 均方根 RMS
        # 对每一个 token 的特征向量独立计算, 也就是沿着最后一维 dim=-1(d_model)计算
        # keepdim 保持维度数不变,rms为 shape [batch_size, seq_len, 1]
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms * self.gain

        # 恢复原有数据类型
        return x_norm.to(in_dtype)

class SwiGLUFFN(nn.Module):
    """
    x → gate_proj(W1) → SiLU ──┐
    x → up_proj(W3) ───────────┴── ⊙ → down_proj(W2) → output

    公式形式:FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
            shape x: (d_model,)
            shape W1,W3 : (d_ff, d_model)
            shape W2: (d_model, d_ff)
    注意: 在Transformer中, 它是 position-wise的(故上面的x是长度为d_model的向量,即一个token)
    """
    def __init__(self, d_model: int,  device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        # d_ff 根据实际情况自动调整,  在(8/3)d_model附近且恰好是64的整数倍
        ffactor = float(8*d_model)/3/64
        factor = int(ffactor) if ffactor-int(ffactor) < 0.5 else int(ffactor) +  1
        self.d_ff = factor*64

        self.weight_1 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.weight_3 = nn.Parameter(torch.Tensor(self.d_ff, self.d_model))
        self.weight_2 = nn.Parameter(torch.Tensor(self.d_model, self.d_ff))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        """
        proj_1 = einsum(x, self.weight_1, "... d_model, d_ff d_model -> ... d_ff")
        silu = proj_1 * torch.sigmoid(proj_1)

        proj_3 = einsum(x, self.weight_3, "... d_model, d_ff d_model -> ... d_ff")

        gate = silu * proj_3

        return einsum(gate , self.weight_2, "... d_ff, d_model, d_ff -> ... d_model")

class RotaryPositionalEmbedding(nn.Module):
    """
    1.名字虽然带有 Embedding, 但与对输入的 input embedding 这种可学习的权重表无关系.
        此处应该更倾向于 encoding... 不要过于纠结名字...
    
    2.RoPE 是在单头的QK上做的, 即输入是分头后的 Q K 
    """
    def __init__(self, theta: float, d_qk: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float  Θ value for the RoPE
        d_qk: int  dimension of query and key vectors. 也就是 d_model 分头后的长度!!!
        max_seq_len: int  Maximum sequence length that will be input
        device: torch.device | None = None  Device to store the buffer on
        """
        super().__init__()
        assert d_qk % 2 == 0, "RoPE requires d_qk to be even"
        self.theta = theta
        self.d_qk = d_qk
        self.max_seq_len = max_seq_len
        # 以下计算规则参考  assignment1文档 3.4.3 


        # 预计算频率(即:   1/theta^((2k-2)/d)    ) 
        # idx = [0,2,4...d_qk), 即 2k-2, k属于 {1,2,... d/2}, 这里 d 即入参 d_qk
        # freqs shape (d_qk/2,)
        idx = torch.arange(0, d_qk, 2)          
        freqs = 1.0 / (self.theta ** (idx / d_qk))

        # 预计算所有 0~max_seq_len-1位置(即position i) 旋转参数 cos sin
        # 即: cos(i*freq)  和  sin(i*freq), 具体如何变化的,详见 forward
        # all_pos shape (max_seq_len,), freqs shape (d_qk/2,)
        #       广播后相乘 theta_ik shape (max_seq_len, d_qk/2)
        # 文档中的旋转矩阵R是一个稀疏矩阵,实际只需计算出每个位置(i,k)的余弦/正弦 值 即可!
        all_pos = torch.arange(0, max_seq_len)
        theta_ik = all_pos[:, None] * freqs[None, :]
        rotate_cos_table = torch.cos(theta_ik)
        rotate_sin_table = torch.sin(theta_ik)

        # 注册buffer,不参与梯度更新, 不保存到state_dict (模型初始化时生成)
        self.register_buffer("rotate_cos_table", rotate_cos_table, persistent=False)
        self.register_buffer("rotate_sin_table", rotate_sin_table, persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor: 
        """
        x (..., seq_len, d_qk): 需要兼容可能不止一个batch维度的情况.
                                注意x是分头后的,所以 d_qk = d_model/h
        token_positions (..., seq_len), 最后一维描述了该 token 在当前token序列中的位置编号pos;
                                 ... 和 x 的... 是对应的
        """
        # rotate_cos_table/rotate_sin_table: (max_seq_len, d_qk/2)
        # token_positions: (..., seq_len). 注意索引规则,索引张量的维度放在前面
        # cos/sin: (..., seq_len, d_qk/2)
        # 较复杂,好好理解这里的索引规则,形状变化
        cos = self.rotate_cos_table[token_positions]
        sin = self.rotate_sin_table[token_positions]

        # 按奇偶拆分. x0/x1 shape (..., seq_len, d_qk/2)
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]

        # 旋转
        rx0 = x0 * cos - x1 * sin
        rx1 = x0 * sin + x1 * cos

        # 合并 y.shape (..., seq_len, d_qk)
        y = torch.stack([rx0, rx1], dim=-1).flatten(-2)
        return y


def softmax(x:torch.Tensor, dim:int) -> torch.Tensor:
    """
    x: (batch_size, seq_len, d_qk)
    dim: 指定用哪一维来计算 softmax
    """
    # 先减去当前维度最大值, 防止指数爆炸溢出. 不改变 softmax 结果(见pdf说明)
    max_vals = torch.max(x, dim=dim, keepdim=True)[0]
    exp_x = torch.exp(x - max_vals)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(Q: torch.Tensor, K:torch.Tensor, V:torch.Tensor,
                                mask:torch.Tensor=None) -> torch.Tensor:
    """
    点积缩放注意力
    Q.shape         (batch_size, ..., seq_len, d_qk)    中间的省略号(...)一般就对应heads的数量
    K.shape         (batch_size, ..., seq_len, d_qk)    Q与K维度/形状相同
    V.shape         (batch_size, ..., seq_len, d_v)
    mask.shape      (seq_len, seq_len)                  => (i,j)为True,表示第i个query需要关注第j个key
    output.shape    (batch_size, ..., seq_len, d_v)
    TODO: 
        1.Q K V 的思想理解!!!! 注意力 的理解!!! 
        2.mask的优势, 什么时候用mask, mask怎么得来的??

        mask的价值: 需要屏蔽某些位置的时候,用来替代循环切片的笨办法. 不需要屏蔽直接不传即可
    """

    # QKt (batch_size, ..., seq_len, seq_len)
    # 第一个seq_len是T_q, 即 query的序列长度;  第二个seq_len是T_k,即 key的序列长度
    qk = einsum(Q, V, "batch_size ... seq_len d_qk, batch_size ... seq_len d_qk -> batch_size ... seq_len seq_len")

    # 缩放因子. dk就是分头后的特征长度 d_qk
    dk = Q.shape[-1]
    scaled_factor = 1.0/torch.sqrt(dk)

    # 历史原因这部分被称作score(叫logits可能更合适).
    # 含义:只是衡量 query 和 key 有多相似. 但是: 权重之和不等于 1
    attn_score = qk/scaled_factor
    if mask is not None:
        attn_score = attn_score.masked_fill(mask == False, -torch.inf)

    # softmax. 注意dim=-1(key). 归一之后,和为 1. 因此被称为 weight
    # 沿着 key 这一维归一, 对每个 query, 所有 key 的注意力权重加起来 = 1
    attn_weight = softmax(attn_score, -1)

    # 含义: 每个query位置,按照它和各个key的相关程度 => 对所有 value 做加权平均
    attn = einsum(attn_weight, V, "batch_size ... seq_len seq_len, batch_size ... seq_len d_v -> batch_size ... seq_len d_v")
    return attn


class CausalMultiHeadSelfAttention(nn.Module):
    """
    因果多头自注意力 MHA
    """
    def __init__(self,d_model:int, num_heads:int, max_seq_len:int, rope_theta:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        # d_q 一定是等于 d_k 的(因此这里统一用d_qk);  但是二者与 d_v 不一定相等
        # 绝大多数标准实现都是 =>   d_q = d_k = d_qk = d_v = d_model/num_heads 
        # 仅少数特殊架构 d_v 取值不等于 d_qk. 
        # 这里按标准实现来!!! 默认 d_model是num_heads的整数倍, 且d_qk是64的整数倍
        self.d_qk = self.d_model / self.num_heads
        self.d_v = self.d_qk
        assert self.d_model % self.num_heads == 0,  "d_model should be multiply of num_heads" 
        assert self.d_qk % 64 == 0 , "d_model/num_heads should be multiply of 64"

        self.weight_Q = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.weight_K = nn.Parameter(torch.empty(self.d_model, self.d_model))
        self.weight_V = nn.Parameter(torch.empty(self.d_model, self.d_model))

        self.weightO = nn.Parameter(torch.empty(self.d_model, self.d_model))

        self.RoPE = RotaryPositionalEmbedding(self.rope_theta, self.d_qk, self.max_seq_len)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x.shape:         (batch_size, seq_len, d_model)
        output.shape:    (batch_size, seq_len, d_model)
        """
        # 1.线性变换得 Q K V 
        # 这里 weight 两个维度其实都是 d_model,若不区分,会对中间那个d_model(逗号旁)求和收缩!!!
        # 最好还是稍作区分,保持 y = xWt 形式
        Q = einsum(x, self.weight_Q, "batch_size seq_len d_model, d_model_out d_model -> batch_size seq_len d_model_out")
        K = einsum(x, self.weight_K, "batch_size seq_len d_model, d_model_out d_model -> batch_size seq_len d_model_out")
        V = einsum(x, self.weight_V, "batch_size seq_len d_model, d_model_out d_model -> batch_size seq_len d_model_out")

        # 2. d_model 拆分为多头(num_heads * d_qk) => 将 num_heads作为一个batch维度; d_qk作为一个更短的特征维度
        #    => shape (batch_size num_heads seq_len d_qk/d_v);  (rearrange同时进行 reshape 和 transpose)
        mh_Q = rearrange(Q, "batch_size seq_len (num_heads d_qk) -> batch_size num_heads seq_len d_qk", num_heads = self.num_heads)
        mh_K = rearrange(K, "batch_size seq_len (num_heads d_qk) -> batch_size num_heads seq_len d_qk", num_heads = self.num_heads)
        mh_V = rearrange(V, "batch_size seq_len (num_heads d_v) -> batch_size num_heads seq_len d_v", num_heads = self.num_heads)

        # 3.对 mh_Q  mh_K 旋转位置编码 RoPE
        mh_Q = self.RoPE(mh_Q)
        mh_K = self.RoPE(mh_K)

        # 4.分头计算 scaled dot-product attention. 支持 causal mask
        # attn.shape (batch_size, num_heads, seq_len, d_v)
        seq_len = x.shape[-2]
        allow_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=0)
        attn = scaled_dot_product_attention(mh_Q, mh_K, mh_V, allow_mask)

        # 5.多头 concat, 然后使用 weight_O 线性变化
        # mh_concat.shape (batch_size seq_len d_model)
        attn_concat = rearrange(attn, "batch_size num_heads seq_len d_v -> batch_size seq_len (num_heads d_v)")
        y = einsum(attn_concat, self.weightO, "batch_size seq_len d_model, d_model_out d_model -> batch_size seq_len d_model_out")
        return y




##################################### test #####################################
def my_try_torch_nn_linear():
    # 1. 定义线性层：y = x @ W.T + b
    # nn.Linear(in_features, out_features, bias=True)
    linear = nn.Linear(in_features=3, out_features=2)

    # 查看权重和偏置
    print("weight shape:", linear.weight.shape)  # [2, 3]  => 转置为 [3,2]
    print("bias shape:", linear.bias.shape)      # [2]

    # 2. 构造输入
    # 输入形状：[batch_size, in_features]
    # batch_size 是数据条数;  in_features是每条数据的特征维度
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])  # batch=2, feature_dim=3

    # 3. 前向计算
    y = linear(x)
    print("output shape:", y.shape)  # [2, 2]
    print(y)

def test_my_Linear():
    linear = Linear(2,3)
    linear.weight = nn.Parameter(
        torch.Tensor(
            [[1, 2, 3],
            [4, 5, 6]]
        )
    )

    x = torch.Tensor(
        # [[1, 0],
        # [0, 1],
        # [1, 2]]

    [[1, 0, 1],
    [0, 1, 2]]

    )

    y = linear.forward(x)
    print(y)



if __name__=="__main__":
    print("\ntest")
    test_my_Linear() 
    