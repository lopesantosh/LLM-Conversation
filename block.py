import torch
import torch.nn as nn

from attention import MultiHeadAttention
from gelu import GELU
from layernorm import LayerNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layernorm1 = LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(d_in = cfg["emb_dim"], 
                                      d_out = cfg["emb_dim"], 
                                      context_length = cfg["context_length"], 
                                      num_heads = cfg["n_heads"],
                                      dropout = cfg["drop_rate"],
                                      qkv_bias = cfg["qkv_bias"])
        self.layernorm2 = LayerNorm(cfg["emb_dim"])
        self.ff = FeedForward(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        # attention component
        shortcut = x
        x = self.layernorm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut

        # feedforward component
        shortcut = x
        x = self.layernorm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
        


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4*cfg["emb_dim"]), 
                                    GELU(), 
                                    nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"]))

    def forward(self, x):
        return self.layers(x)


        


if __name__ == '__main__':
    cfg = {"vocab_size": 50257, 
           "context_length": 1024, 
           "emb_dim": 768, 
           "n_heads": 12, 
           "n_layers": 12, 
           "drop_rate" : 0.1, 
           "qkv_bias": False}
    
    x = torch.rand(2, 4, 768)
    block = TransformerBlock(cfg)
    output = block(x)
    print("Input shape: ", x.shape)
    print("Output shape: ", output.shape)
    
