"""
let's implement a simple attention mechanism with training parameters, 
which called scaled dot-product attention.
This code is based on the previous simple attention code,
but now we will use learnable parameters for the query, key, and value vectors.

W_query, W_key, and W_value are weight matrices for the query, key, and value vectors.

Q=X @ W_query
K=X @ W_key
V=X @ W_value

Attention(Q,K,V)=softmax( QK^T/sqrt(d_k) )V

"""
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(f"{query_2=}")
print(f"{key_2=}")
print(f"{value_2=}")


keys=inputs @ W_key
values=inputs @ W_value
print(f"keys.shape={keys.shape}")
print(f"{values.shape=}")

# atten_score \omega_{22} = q_2 \cdot k_2^T
# 这里的 q_2 和 k_2 是通过输入 x_2 计算
# query_2 和 key_2 的点积得到的
keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(f"{attn_scores_22=}")

attn_scores_2 = query_2 @ keys.T
print(f"{attn_scores_2=}")

q_k = keys.shape[1]
attn_scores_2 = attn_scores_2 / (q_k ** 0.5)
attn_weights_2 = torch.softmax(attn_scores_2, dim=-1)
print(f"{attn_weights_2=}")
context_vec_2 = attn_weights_2 @ values
print(f"{context_vec_2=}")
# context_vec_2=tensor([0.2854, 0.4081])

