"""
let 's implement a simple attention mechanism without training parameters,
which is called simple attention.


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


query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(f"{attn_scores_2=}")

# 归一化
print(f"========加权归一jj=======")
attn_scores_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("ATTENTION WIGHTS :", attn_scores_2_tmp)
print(f"SUM: {attn_scores_2_tmp.sum()}")

def softmax_native(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

print(f"=========softmax 归一=====")
attn_scores_2_native = softmax_native(attn_scores_2)
print("ATTENTION WIGHTS :", attn_scores_2_native)
print(f"SUM: {attn_scores_2_native.sum()}")

print(f"=========softmax 归一=====")
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("ATTENTION WIGHTS :", attn_weights_2)
print(f"SUM: {attn_weights_2.sum()}")

print(f"为解决上下文中的同词不同意现象，，需要通过上下文内容对语义空间做投影")
print(f"""上下文向量 = \simga_i attn_weight_i * word_embd_i""")
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(f"{query=}")
print(f"{context_vec_2=}")

print(f"========拓展计算过程至批量计=======")
atte_scores = torch.empty((inputs.shape[0], inputs.shape[0]))
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        atte_scores[i][j] = torch.dot(x_i, x_j)
print(f"{atte_scores=}")

"""使用矩阵计算
  X = [x^1, x^2, x^3, x^4, x^5, x^6] 
{6*3}

atte_scores = X * X^T
{6*6}
atte_weights = softmax(atte_scores)
context_vec = sigma_i atte_weights_i * x^i
"""

print(f"========使用矩阵计算=======")
atte_scores_mat = torch.matmul(inputs, inputs.T)
print(f"{atte_scores_mat=}")
attn_weights = torch.softmax(atte_scores_mat, dim=-1)
print(f"{attn_weights=}")
print(f"all row sums: {attn_weights.sum(dim=-1)}")
all_context_vecs = torch.matmul(attn_weights, inputs)
print(f"{all_context_vecs=}")

print(f"Previous 2nd context vector: {context_vec_2=}")
