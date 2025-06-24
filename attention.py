import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, is_causal=True, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.is_causal = is_causal  # True for causual attention
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        
        if(is_causal):
            # stricly upper triangualr matrix with 1s
            self.register_buffer("mask",
                             torch.triu(torch.ones(context_length, context_length), diagonal = 1)
                            )
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)  # shape (b, num_tokens, d_out)
        
        # change shape to (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        
        # transpose from (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # compute dot product for each head
        att_scores = queries @ keys.transpose(2, 3)
        
        # apply mask if mask_bool is true
        if(self.is_causal):
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens] # truncate to num_tokens
            att_scores.masked_fill_(mask_bool, -torch.inf)         # replace mask entry by -inf
        
        # normalize and apply dropout
        att_weights = torch.softmax(att_scores / keys.shape[-1]**0.5, dim = -1)
        att_weights = self.dropout(att_weights)
        
        # context vectors AV
        context_vec = att_weights @ values
        
        # transpose from (b, num_heads, num_tokens, head_dim) to (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        
        # bring shape to (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        
        # add optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec
    
     

if __name__ == '__main__':
    inputs = torch.tensor([[0.43, 0.15, 0.89], [0.55, 0.87, 0.66], [0.57, 0.85, 0.64], [0.22, 0.58, 0.33], 
                           [0.77, 0.25, 0.10], [0.05, 0.80, 0.55]])
    batch = torch.stack((inputs, inputs, inputs), dim=0)
    print(batch.shape)
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    att = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vecs = att(batch)
    print(context_vecs)
    print(context_vecs.shape)
                          
    
        
        
                             
        