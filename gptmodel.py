import os
import torch
import torch.nn as nn
import numpy as np

from block import TransformerBlock
from layernorm import LayerNorm
from load_model import load_gpt2
from configs import get_config

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential( *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def get_model(model_name, base_dir):
    cfg = get_config(model_name)
    model = GPTModel(cfg)

    # get pretrained parameters
    s = model_name.split("-")
    model_dir = os.path.join(base_dir, s[0], s[1])
    settings, params = load_gpt2(model_dir)

    # load pretrained parameters into a model
    load_weights(model, params)
    return model

    
def load_weights(gpt, params):
    
    # position and token embedding weight
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        
        # divide attention weights into three equal parts for query, key, value
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)        
        gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        #divide attention bias into three equal parts for query, key, value
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

        # out projection weight and bias
        gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight,
                                                       params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias,
                                                     params["blocks"][b]["attn"]["c_proj"]["b"])

        # feed forward layers weight and bias
        gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight,
                                                       params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias,
                                                     params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight,
                                                       params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias,
                                                     params["blocks"][b]["mlp"]["c_proj"]["b"])

        # norm1 scale and shift
        gpt.trf_blocks[b].layernorm1.scale = assign(gpt.trf_blocks[b].layernorm1.scale, params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].layernorm1.shift = assign(gpt.trf_blocks[b].layernorm1.shift, params["blocks"][b]["ln_1"]["b"])

        # norm2 scale and shift 
        gpt.trf_blocks[b].layernorm2.scale = assign(gpt.trf_blocks[b].layernorm2.scale, params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].layernorm2.shift = assign(gpt.trf_blocks[b].layernorm2.shift,params["blocks"][b]["ln_2"]["b"])

    # final norm scale and shift
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    
    # outer layer weight
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


if __name__ == "__main__":
    import tiktoken

    model_name = "gpt2-medium"
    base_dir = "/Users/santosh/Documents/workspace/models"
    model = get_model(model_name, base_dir)
    
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(text1)))
    batch.append(torch.tensor(tokenizer.encode(text2)))
    batch = torch.stack(batch, dim = 0)
    out = model(batch)
    print("Input batch: \n", batch)
    print("Output shape: \n", out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    size = total_params *4 /(1024 * 1024)
    print("Total number of parameters: ", total_params)
    print(f"Total size of the model:  {size: 0.2f} MB")

    

    
    
        
        
        
        

