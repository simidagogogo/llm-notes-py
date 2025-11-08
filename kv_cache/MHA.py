# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input, mask=None):
        """
        把mask作为参数传入attention, 是深度学习工程中的最佳实践: 
        a) 灵活支持train/inference场景: 训练时一般需要mask, 推理时(尤其是单步自回归)通常不用mask
        b) 它让attention模块变得通用、灵活、易于扩展和复用, 适配各种不同的实际任务和需求. 例如: 

        # 1. 自回归因果mask(下三角为True, 屏蔽未来)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).unsqueeze(0)    # [1, seq_len, seq_len]

        # 2. Padding mask
        # 假设样本1有效长度为3, 样本2有效长度为2
        lengths = torch.tensor([3, 2])
        padding_mask = torch.arange(seq_len).expand(batch_size, seq_len) < lengths.unsqueeze(1)  # [batch, seq_len], True=有效token

        # 转成attention mask: [batch, 1, seq_len]
        attn_pad_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)                        # [batch, seq_len, seq_len]

        # 3. 因果mask与padding mask结合
        # 先下三角(因果), 再加padding mask
        full_mask = causal_mask & attn_pad_mask                                                  # [batch, seq_len, seq_len]
        
        @input: (batch, seq_len_so_far, embed_dim)
        """
        queries = self.query_proj(input)            # (batch, seq_len_so_far, embed_dim)
        keys = self.key_proj(input)                 # (batch, seq_len_so_far, embed_dim)
        values = self.value_proj(input)             # (batch, seq_len_so_far, embed_dim)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / keys.size(-1) ** 0.5
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_probs = self.softmax(attn_scores)      # (batch, seq_len_so_far, seq_len_so_far)
        return torch.matmul(attn_probs, values)     # (batch, seq_len_so_far, embed_dim)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SimpleGPTModel(nn.Module):
    def __init__(self, vocab_size, emb_size, max_seq_len=512):
        super().__init__()
        
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(max_seq_len, emb_size)
        self.attention = SimpleAttention(emb_size)
        self.feed_forward = FeedForward(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)

    def forward(self, idx):
        # 1. 将token转换为emb
        token_embeddings = self.embedding_layer(idx)                   # [batch_size, seq_len, emb_size]
        
        # 2. 位置编码
        seq_len = idx.size(1)
        positions = torch.arange(seq_len).unsqueeze(0)                 # [1, seq_len]
        pos_embeddings = self.position_embedding(positions)            # [1, seq_len, emb_size]
        pos_embeddings = pos_embeddings.expand(idx.size(0), -1, -1)    # [batch_size, seq_len, emb_size]
        
        # 3. 最终emb
        all_embeddings = token_embeddings + pos_embeddings             # [batch_size, seq_len, emb_size]
        
        # 4. 经过transformer block
        # 每步都把全部历史token送入attention重复计算所有key/value -> 在长序列生成时效率很低
        output = self.attention(all_embeddings)                        # [batch_size, seq_len, emb_size]
        output = self.feed_forward(output)                             # [batch_size, seq_len, emb_size]

        # 5. 计算logits
        logits = self.lm_head(output)                                  # [batch_size, seq_len, vocab_size]
        return logits


# A function for the GPT model to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens - 1):
        idx_cond = idx[:, -context_size:]                           # 这里需要输入全部上下文
        
        # 1. 计算logits
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 2. 计算probs. 仅取最后一个位置logits(在自回归生成时只关心最后一个token输出)
        probs = torch.softmax(logits[:, -1, :], dim=-1)             # [batch_size, vocab_size]

        # 3. 从概率分布中采样,获得下一个 token
        # 3.1 贪心解码(取最大概率token): 容易陷入循环、缺乏多样性
        next_token = torch.argmax(probs, dim=-1, keepdim=True)      # [batch_size, 1]
        
        # 4. 自回归方式构建tokens
        idx = torch.cat((idx, next_token), dim=1)                   # [batch_size, max_new_tokens]
    return idx


torch.manual_seed(42)
vocab_size = 65535
max_new_tokens = 100 # 调小点可以很快出结果
context_size = 1024
batch_size = 32
model = SimpleGPTModel(vocab_size=vocab_size, emb_size=16, max_seq_len=context_size)
idx = torch.randint(0, vocab_size, (batch_size, 1))  # [batch_size, 1]
output_idx = generate_text_simple(model, idx, max_new_tokens, context_size)
print(f"output_idx.shape: %{output_idx.shape}")
print(f"output_idx[0, :10]: %{output_idx[0, :10]}")
