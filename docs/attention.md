# 自注意力机制（Self-Attention）原理与实现

自注意力机制是Transformer模型的核心组件，用于捕捉序列中元素间的依赖关系。其核心思想是通过计算序列中每个元素与其他元素的关联权重（注意力分数），然后基于权重对值（Value）进行加权聚合。

---

## 1. 数学原理

### 输入表示
输入序列为 $X \in \mathbb{R}^{n \times d}$，其中：
- $n$：序列长度
- $d$：特征维度

### 线性变换
通过可学习权重矩阵生成**Query**、**Key**和**Value**：
$$
\begin{aligned}
Q &= X W_Q \quad (W_Q \in \mathbb{R}^{d \times d_k}) \\
K &= X W_K \quad (W_K \in \mathbb{R}^{d \times d_k}) \\
V &= X W_V \quad (W_V \in \mathbb{R}^{d \times d_v})
\end{aligned}
$$

### 注意力计算
$$
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

---

## 2. 公式分步解析

### （1）$QK^T$ 计算
- 输出矩阵 $A = QK^T \in \mathbb{R}^{n \times n}$
- $A_{ij}$ 表示第 $i$ 个元素与第 $j$ 个元素的相似度

### （2）缩放因子 $\sqrt{d_k}$
- 防止 $d_k$ 较大时点积值过大导致梯度消失

### （3）Softmax归一化
$$
\text{softmax}(A_{ij}) = \frac{e^{A_{ij}}}{\sum_{k=1}^n e^{A_{ik}}}
$$

### （4）加权聚合
$$
\text{Output}_i = \sum_{j=1}^n \text{softmax}(A_{ij})V_j
$$

---

## 3. 物理意义
| 符号 | 含义 | 类比 |
|------|------|------|
| $Q$ | 当前需要计算注意力的位置 | "我在看哪里" |
| $K$ | 其他位置的信息 | "我在看什么" |
| $V$ | 实际被聚合的信息 | "我看到的内容" |
| $\text{softmax}$ | 注意力权重归一化 | 聚焦重要部分 |

---
## 4. Python实现（PyTorch）
```python
import torch
import torch.nn.functional as F

def self_attention(X, d_k):
    # X: [batch_size, seq_len, d_model]
    Q = X @ W_Q  # [batch_size, seq_len, d_k]
    K = X @ W_K  # [batch_size, seq_len, d_k] 
    V = X @ W_V  # [batch_size, seq_len, d_v]
    
    scores = (Q @ K.transpose(-2, -1)) / (d_k**0.5)
    attn_weights = F.softmax(scores, dim=-1)
    return attn_weights @ V

```
