import torch
import torch.nn as nn

class MultHeadAttention(nn.Module):
    def __init(self, d_in, d_out,
               context_length, num_heads=8, 
               dropout=0.5, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, \
            "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(p=dropout)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)  # (b, num_tokens, d_out)
        queries = self.W_query(x) # (b, num_tokens, d_out)
        values  = self.W_value(x) # (b, num_tokens, d_out)

        queries = queries.view(
            b, num_tokens, self.num_heads, self.head_dim
        )
        keys = keys.transpose(1, 2)
        queries =queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)      # (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
            attn_scores / (self.head_dim**0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1, 2)  # (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )
        context_vec = self.out_proj(context_vec)  # (b, num_tokens, d_out)
        return context_vec

if __name__ == "__main__":
    gpt2 = MultHeadAttention(
        d_in=768, d_out=768, context_length=1024,
        num_heads=12, dropout=0.1, qkv_bias=True
    )