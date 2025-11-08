# -*- coding: utf-8 -*-

from typing import Optional
import torch
import torch.nn as nn


class SimpleAttentionKVCache(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, key_cache, value_cache, mask=None):
        # input_token: (batch, 1, embed_dim)
        k_new = self.key_proj(input)                # (batch, 1, embed_dim)
        v_new = self.value_proj(input)              # (batch, 1, embed_dim)
        q_new = self.query_proj(input)              # (batch, 1, embed_dim)
        
        # 拼接历史缓存
        keys = torch.cat([key_cache, k_new], dim=1) if key_cache is not None else k_new         # (batch, seq_len_so_far, embed_dim)
        values = torch.cat([value_cache, v_new], dim=1) if value_cache is not None else v_new   # (batch, seq_len_so_far, embed_dim)
        
        attn_scores = torch.matmul(q_new, keys.transpose(-2, -1)) / keys.size(-1) ** 0.5        # (batch, 1, seq_len_so_far)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        attn_probs = self.softmax(attn_scores)                                                  # (batch, 1, seq_len_so_far)
        output = torch.matmul(attn_probs, values)                                               # (batch, 1, embed_dim)
        return output, keys, values


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


class SimpleGPTModelKVCache(nn.Module):
    def __init__(self, vocab_size, emb_size, context_size=512):
        """
        @context_size: 即max_seq_len
        """
        super().__init__()
        self.context_size = context_size
        self.embedding_layer = nn.Embedding(vocab_size, emb_size)
        self.position_embedding = nn.Embedding(context_size, emb_size)
        self.attention = SimpleAttentionKVCache(emb_size)
        self.feed_forward = FeedForward(emb_size)
        self.lm_head = nn.Linear(emb_size, vocab_size)
        self.key_cache = None
        self.value_cache = None
        
    def forward(self, idx):
        """
        @idx: [batch_size, 1]
        """
        # print(f"idx.shape: {idx.shape}")
        # 1. 将token转换为emb
        token_embeddings = self.embedding_layer(idx)                   # [batch_size, 1, emb_size]
        
        # 2. 位置编码
        seq_len = idx.size(1)
        positions = torch.arange(seq_len).unsqueeze(0)                 # [1, 1]
        pos_embeddings = self.position_embedding(positions)            # [1, 1, emb_size]
        pos_embeddings = pos_embeddings.expand(idx.size(0), -1, -1)    # [batch_size, 1, emb_size]
        
        # 3. 最终emb
        all_embeddings = token_embeddings + pos_embeddings             # [batch_size, 1, emb_size]
        
        # 4. 经过transformer block
        # KV缓存: 显著减少重复计算, 推理效率提升
        # 每步只对当前新token做attention计算key/value, 历史部分直接从缓存读取拼接, 无需重复计算
        # output:      [batch_size, 1, emb_size]
        # key_cache:   [batch_size, seq_len, emb_size]
        # value_cache: [batch_size, seq_len, emb_size]
        
        self.key_cache = self.key_cache[:, -(self.context_size-1):, :] if self.key_cache is not None else None
        self.value_cache = self.value_cache[:, -(self.context_size-1):, :] if self.value_cache is not None else None

        output, self.key_cache, self.value_cache = self.attention(all_embeddings, self.key_cache, self.value_cache)
        # print("self.key_cache.shape=", self.key_cache.shape) 
        output = self.feed_forward(output)                             # [batch_size, 1, emb_size]

        # 5. 计算logits
        logits = self.lm_head(output)                                  # [batch_size, 1, vocab_size]
        return logits

    
# A function for the GPT model to generate text
def generate_text_simple(model, idx, max_new_tokens):
    for _ in range(max_new_tokens - 1):
        idx_cond = idx[:, -1:]                                         # 这里每次仅需要输入一个最新的token 
        # 1. 计算logits
        with torch.no_grad():
            logits = model(idx_cond)
        
        # 2. 计算probs. 仅取最后一个位置logits(在自回归生成时只关心最后一个token输出)
        probs = torch.softmax(logits[:, -1, :], dim=-1)                # [batch_size, vocab_size]

        # 3. 从概率分布中采样,获得下一个 token
        # 3.1 贪心解码(取最大概率token): 容易陷入循环、缺乏多样性
        next_token = torch.argmax(probs, dim=-1, keepdim=True)         # [batch_size, 1]
        
        # 4. 自回归方式构建tokens
        idx = torch.cat((idx, next_token), dim=1)                      # [batch_size, max_new_tokens]
    return idx


torch.manual_seed(42)
vocab_size = 65535
max_new_tokens = 100 # 调小点可以很快出结果
context_size = 1024
batch_size = 32
model = SimpleGPTModelKVCache(vocab_size=vocab_size, emb_size=16, context_size=context_size)
idx = torch.randint(0, vocab_size, (batch_size, 1))  # [batch_size, 1]
output_idx = generate_text_simple(model, idx, max_new_tokens)
print(f"output_idx.shape: %{output_idx.shape}")
print(f"output_idx[0, :10]: %{output_idx[0, :10]}")
