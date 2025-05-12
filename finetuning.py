import json
import os
import re
import time
import urllib
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from functools import partial

from gpt_download import download_and_load_gpt2

from gpt import (
    GPTModel,
    load_weights_into_gpt,
    calc_loss_loader,
    train_model_simple,
    text_to_token_ids,
    token_ids_to_text, 
    generate
)


class DialogueDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            instruction_plus_input = format_input(entry)
            entries = re.split(r'Alex|Bob', entry)
            bob_response_text = f"\n\n### Bob:\n{entries[2][4:-4]}" if entries[2] else ""
            alex_response_text = f"\n\n### Alex:\n{entries[3][4:-1]}" if entries[3] else ""
            response_text = bob_response_text + alex_response_text
            
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    
    def __len__(self):
        return len(self.data)
    

def download_and_load_file(file_path, url):
    """ load dataset from file or url. """
    
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    
    
    f = open('llm_train_text.txt')
    data = f.readlines()
    
    # clean text
    for i in range(len(data)):
        entry = data[i]
        data[i] = entry[1:-2]
                                   
    return data    


def format_input(entry):
    """ format data entry to Alpaca prompt style. """
    
    entries = re.split(r'Alex|Bob', entry)
    instruction_text = (
        f"Below is a text that describes a conversation between two people. "
        f"Write a response that appropriately follow the conversation."
        f"\n\n### Alex:\n{entries[1][4:-4]}"
    )

    return instruction_text


def split_train_test(data, frac=0.7):
    """ split data in training and validation set. """
    
    train_portion = int(len(data) * frac)  # percentage for training

    train_data = data[:train_portion]
    val_data = data[train_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print(50*"-")
    
    return train_data, val_data
    
    
    
def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    """ crate custom collate function to handle specific requirements and formating of dialogue dataset. """
    
    # find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # add an <|endoftext|> token
        new_item += [pad_token_id]
        # pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # shift +1 to the right for targets

        # replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def main(test_mode=False):
    
    torch.manual_seed(123)
    
    #######################################
    # Download and prepare dataset
    #######################################
    file_path = "llm_train_text.json"
    url = None
    data = download_and_load_file(file_path, url)
    train_data, val_data = split_train_test(data, frac=0.7)

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # create a new version of the function
    collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    num_workers = 0
    batch_size = 2

    train_dataset = DialogueDataset(train_data, tokenizer)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              collate_fn=collate_fn, 
                              shuffle=True,
                              drop_last=True, 
                              num_workers=num_workers
                             )

    val_dataset = DialogueDataset(val_data, tokenizer)
    val_loader = DataLoader(val_dataset, 
                            batch_size=batch_size,
                            collate_fn=collate_fn,
                            shuffle=False, 
                            drop_last=False,
                            num_workers=num_workers
                           )
    
    
    #######################################
    # Load pretrained model
    #######################################


    # using GPT model for training
    BASE_CONFIG = {"vocab_size": 50257,     # Vocabulary size
                   "context_length": 1024,  # Context length
                   "drop_rate": 0.0,        # Dropout rate
                   "qkv_bias": True         # Query-key-value bias
                  }

    model_configs = {"gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
                     "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
                     "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
                     "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
                    }

    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################
    num_epochs = 10
    
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=2)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=2)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    # use adamw optimizer, learning weight and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    # training loops
    train_losses, val_losses, tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device,                       num_epochs=num_epochs, eval_freq=2, eval_iter=5, start_context=format_input(val_data[0]), tokenizer=tokenizer
                                                              )
    print(50*"-")

    #######################################
    # Evaluating and Saving results
    #######################################
    print("Generating conversational responses")
    
    result = [None] * len(val_data)
    for i, entry in tqdm(enumerate(val_data), total=len(val_data)):
        result[i] = {}
        
        s = re.split(r'Alex|Bob', entry)
        alex_input = f"Alex:{s[1][4:-4]}"
        bob_response = f"Bob: {s[2][4:-4]}"
        alex_response= f"Alex: {s[3][4:-1]}"
        true_response = bob_response + alex_response
        
        
        input_text = format_input(entry)
        token_ids = generate(model=model,
                             idx=text_to_token_ids(input_text, tokenizer).to(device),
                             max_new_tokens=50,
                             context_size=BASE_CONFIG["context_length"],
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
        
        print(alex_input)
        print('True Response: {0}'.format(true_response))
        print('Generated Response: {0}'.format(generated_response))
        print()
        

    with open("dialogue-data-with-gen-response.json", "w") as file:
        json.dump(result, file, indent=4)  # "indent" for pretty-printing


if __name__ == '__main__':
    
    main()