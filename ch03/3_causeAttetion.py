import torch
from attenion_class import  SelfAttention_v2
from cont import inputs

torch.manual_seed(789)
sa_v2= SelfAttention_v2(d_in=3, d_out=2)
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T 
attn_weights = torch.softmax(attn_scores / (keys.shape[-1]**0.5), dim=-1)

print(f"原始注意力分{attn_weights=}")

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(f"mask 矩阵为{mask_simple=}")

masked_simple = attn_weights * mask_simple
print(f"掩码后的矩阵{masked_simple=}")

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(f"归一化后的矩阵{masked_simple_norm=}")

print("========softmax 归一化============")
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), float('-inf'))
print(masked)
attn_weights = torch.softmax(masked / (keys.shape[-1]**0.5), dim=1)
print(f"attn_weights={attn_weights}")

print(f"======== dropout ====")
torch.manual_seed(123)
dropout = torch.nn.Dropout(p=0.5)
example = torch.ones(6, 6)
print(dropout(example))

