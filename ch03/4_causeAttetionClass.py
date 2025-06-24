import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,
                 dropout, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

    def forward(self, x):
        k, num_tokens, d_in = x.shape
        keys    = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values  = self.W_value(x) # (b, num_tokens, d_out)

        attn_scores = queries @ keys.transpose(1, 2)  # (b, num_tokens, num_tokens)
        print(f"{attn_scores=}")
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf 
        )
        attn_weights = torch.softmax(
            attn_scores / (keys.shape[-1]**0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values  # (b, num_tokens, d_out)
        return context_vec

class MultHeadCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads=8, dropout=0.5, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(
                d_in, d_out, context_length, dropout, qkv_bias
            ) for _ in range(num_heads)
        ])
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

if __name__ == "__main__":
    torch.manual_seed(123)
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89], # Your     (x^1)
         [0.55, 0.87, 0.66], # journey  (x^2)
         [0.57, 0.85, 0.64], # starts   (x^3)
         [0.22, 0.58, 0.33], # with     (x^4)
         [0.77, 0.25, 0.10], # one      (x^5)
         [0.05, 0.80, 0.55]] # step     (x^6)
    )  # Add batch dimension

    batch = torch.stack((inputs, inputs), dim=0)  # Create a batch of two identical inputs
    print(batch.shape)

    print(f"================单头因果注意力===============")
    context_length = batch.shape[1]
    ca = CausalAttention(d_in=3, d_out=2, context_length=context_length, dropout=0.5)
    print(ca(batch)) 

    print(f"================多头因果注意力===============")
    mha = MultHeadCausalAttention(d_in=3, d_out=2, context_length=context_length, num_heads=2, dropout=0.5)
    print(mha(batch))
