

def get_config(model_name = "gpt2-small"):
    """ This function define the model config for the gpt2 model. """
    
    gpt_config = {"vocab_size": 50257,      # vocabulary size
                  "context_length": 1024,   # max number of tokens in a text
                  "emb_dim": 768,           # embedding dimension
                  "n_heads": 12,            # number of attention heads
                  "n_layers": 12,           # number of layers
                  "drop_rate" : 0.1,        # dropout rate
                  "qkv_bias": False         # query key value bias
                 }   

    model_configs = {
        "gpt2-small" : {"emb_dim": 768, "n_heads": 12, "n_layers": 12},       # 124M
        "gpt2-medium" : {"emb_dim": 1024, "n_heads": 16, "n_layers": 24},     # 355M
        "gpt2-large": {"emb_dim": 1280, "n_heads": 20, "n_layers": 36},       # 774M
        "gpt2-xl" : {"emb_dim": 1600, "n_heads": 25, "n_layers": 48}          # 1558M
    }


    cfg = gpt_config.copy()
    cfg.update(model_configs[model_name])
    cfg.update({"qkv_bias": True})          # update query key value bias to True

    return cfg

if __name__ == "__main__":
    cfg = get_config("gpt2-medium")
    print(cfg)
    
        
    