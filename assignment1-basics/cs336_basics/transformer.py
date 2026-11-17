import torch
import torch.nn as nn
import nn_module

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int,
                        rope_theta:float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta

        # 网络模块
        self.norm = nn_module.RMSNorm(self.d_model)
        self.multiHeadAttn =  nn_module.CausalMultiHeadSelfAttention(self.d_model,
                                                        self.num_heads,
                                                        self.max_seq_len,
                                                        self.rope_theta
                                                        )
        self.ffn = nn_module.SwiGLUFFN(self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = y.shape  (batch_size, seq_len, d_model)
        """
        y_attn = x + self.multiHeadAttn(self.norm(x))
        y = y_attn + self.ffn(self.norm(y_attn))
        return y



class TransformerLM(nn.Module):
    """
    Language Model
    """
    def __init__(self, d_model:int, num_heads:int, max_seq_len:int,
                        rope_theta:float, vocab_size:int, num_layers):
        super().__init__()
        self.num_layers = num_layers

        self.embedding = nn_module.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, max_seq_len, rope_theta)
                                   for _ in range(num_layers)
        ])
        self.norm = nn_module.RMSNorm(d_model)
        self.linear = nn_module.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape: (batch_size, seq_len), 系 原始句子经过 tokenizer 处理后的数据
        y.shape: (batch_size, seq_len, vocab_size)

        TODO: 输出的 y 如何最终转换为句子, 待跟进
        """
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        x = self.linear(x)
        y = nn_module.softmax(x, dim = -1)
        return y


if __name__=="__main__":
    print("\ntest")


