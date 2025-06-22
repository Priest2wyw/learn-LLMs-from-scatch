# 注意力机制的二次复杂度分析

注意力机制的**二次复杂度**（Quadratic Complexity）是指其计算成本与输入序列长度$n$呈$O(n^2)$关系。以下是详细原因：

---

## 1. 核心计算步骤的复杂度
### （1）$QK^T$ 矩阵乘法
- **输入矩阵**：$Q \in \mathbb{R}^{n \times d}$, $K \in \mathbb{R}^{n \times d}$
- **计算过程**：每个$Q_i$需要与所有$K_j$做点积
- **计算量**：$n \times n$次点积运算 → $O(n^2 d)$

### （2）Softmax归一化
- **输入矩阵**：$A \in \mathbb{R}^{n \times n}$
- **计算过程**：对每行的$n$个元素做指数和归一化
- **计算量**：$n \times n$次指数运算 → $O(n^2)$

### （3）加权求和 $AV$
- **输入矩阵**：$A \in \mathbb{R}^{n \times n}$, $V \in \mathbb{R}^{n \times d}$
- **计算量**：$n \times n \times d$次乘加 → $O(n^2 d)$

---

## 2. 可视化复杂度来源
```text
序列长度 n=4 时的注意力矩阵示例：
       Key1 Key2 Key3 Key4
Query1  •    •    •    •  
Query2  •    •    •    •  
Query3  •    •    •    •  
Query4  •    •    •    •  

```
每个•代表1次相似度计算

总计算量：$n \times n = 16$次

操作	复杂度	说明
全连接层	$O(n d^2)$	权重矩阵与输入相乘
RNN	$O(n d^2)$	时间步的串行处理
自注意力	$O(n^2 d)$	所有位置两两交互


https://blog.csdn.net/shizheng_Li/article/details/144546011