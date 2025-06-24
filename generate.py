from prepare_data import InstructionDataset

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    idx = (batch, n_token) array of indices. 
    max_new_tokens = number of iterations next-token is generated.
    context_size = context_length. 
    """
    
    for _ in range(max_new_tokens):
        idx_cond = [:, -context_size:] # crop if it exceeds the supported context_size
        
        with torch.no_grad():
            logits = model(idx_cond) # gives (batch, n_token, vocab_size)
        
        # focus on the last time step.
        logits = logits[:, -1, :] #(batch, n_token, vocab_size) becomes (batch, vocab_size)
        
        # filter logits with top_k sampling
        if top_k is not None:
            top_logits, top_pos = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float('-inf')).to(logits.device),
                                 logits
                                )
        
        # apply temperature scaling
        if temperature > 0.0:
            logits = logits/temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1) # idx_next has shape (batch,1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True) # greedy next token selection 
        
        # stop generating if end of the sequence token is encountered
        if idx_next == eos_id:
            break
            
        idx = torch.cat((idx, idx_next), dim = 1)      
    return idx


def generate_response(model, val_loader, ):
    
    result = [None] * len(val_data)
    for i, entry in tqdm(enumerate(val_data), total=len(val_data)):
        result[i] = {}
        true_response = entry['output']
        input_text = InstructionDataset.format_input(entry)
        token_ids = generate(model=model,
                             idx=text_to_token_ids(input_text, tokenizer).to(device),
                             max_new_tokens=50,
                             context_size=cfg["context_length"],
                             eos_id=50256
                            )
        
        # generate returns combined input and output. extract the output response
        generated_text = token_ids_to_text(token_ids, tokenizer)
        s = generated_text[len(input_text):].replace("###", "").strip()
        s = s.split()
        generated_response = " ".join([item.strip() for item in s])
        
        result[i]["input"] = alex_input
        result[i]["response"] = true_response
        result[i]["model_response"] = generated_response
        
        
        print('True Response: {0}'.format(true_response))
        print('Generated Response: {0}'.format(generated_response))
        print()

    return result
        

if __name__ == '__main__':
    

    
    
    
       
                                 