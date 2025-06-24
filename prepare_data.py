import json
import os
import re
import urllib
import tiktoken
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PrepareDataset():
    def __init__(self, file_path, url, batch_size, num_workers):
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        #self.data = self.load_file(file_path, url)
        self.data = self.load_dialoguefile(file_path)
        self.partition_data()
        self.tokenizer = tiktoken.get_encoding("gpt2")
        
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
    
    
    def load_file(self, file_path, url):
        """ load or download file and read json data. 
        Return text data. """
        
        # if file_path does not exits
        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")
            f = open(file_path, "w", encoding="utf-8")
            f.write()
    
        file = open(file_path, "r")
        data = json.load(file)
        return data

    
    def load_dialoguefile(self, file_path):
        data = []
        f = open(file_path, 'r')
        s = f.read()
        texts = s.split("{")
        texts = texts[1:]
        
        for text in texts:
            s = re.sub(r'([\n\t}])', r'', text)
            s = re.split(r',\s\s', s)
            
            # make a dictionary and append it to data
            entry = {}
            entry['instruction'] = ""
            entry['input'] = s[0]
            entry['output'] = ' '.join(s[1:]).strip()
            data.append(entry)
        
        return data
            
            
            
        
    
    def partition_data(self, split = 0.7):
        """ partition data into train, val, and test sets. """
        
        train_portion = int(len(self.data) * split)
        val_portion = int(len(self.data) * (1-split))
        test_portion = len(self.data) - train_portion - val_portion
        
        self.train_data = self.data[:train_portion]
        self.val_data = self.data[train_portion:train_portion + val_portion]
        self.test_data = self.data[train_portion+val_portion:]
        return
    
    def create_dataloaders(self):
        
        # train data loader
        train_dataset = InstructionDataset(self.train_data, self.tokenizer)
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size = self.batch_size, 
                                      collate_fn = self.collate_fn, 
                                      shuffle = True, 
                                      drop_last = False, 
                                      num_workers = self.num_workers)
        
        
        # validation data loader
        val_dataset = InstructionDataset(self.val_data, self.tokenizer)
        val_dataloader = DataLoader(val_dataset, 
                                      batch_size = self.batch_size, 
                                      collate_fn = self.collate_fn, 
                                      shuffle = False, 
                                      drop_last = False, 
                                      num_workers = self.num_workers)
        
        
        # test data loader
        test_dataset = InstructionDataset(self.test_data, self.tokenizer)
        test_dataloader = DataLoader(test_dataset, 
                                      batch_size = self.batch_size, 
                                      collate_fn = self.collate_fn, 
                                      shuffle = False, 
                                      drop_last = False, 
                                      num_workers = self.num_workers)
       
        
        print("Train loader: ")
        for inputs, targets in train_dataloader:
            print(inputs.shape, targets.shape)
              
        return train_dataloader, val_dataloader, test_dataloader
        
        


    def collate_fn(self, batch, pad_token_id = 50256, ignore_index = -100, allowed_max_length = None, device = "cpu"):
        """ create specific formating for a batch. 
        Return input and target tensors. """
    
        batch_max_length = max(len(item)+1 for item in batch)
        inputs_list, targets_list = [], []
    
        for item in batch:
            new_item = item.copy()
            new_item += [pad_token_id] # that is why max+1 used 
        
            # pad sequence to max length
            padded = (new_item + (batch_max_length - len(new_item)) * [pad_token_id])
            inputs = torch.tensor(padded[:-1])
            targets = torch.tensor(padded[1:])  # shift one position to the right
        
            mask = targets == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if(indices.numel() > 1):
                targets[indices[1:]] = ignore_index # replace all but first pad token with ignore_index
        
            # truncate to max sequence length
            if allowed_max_length is not None:
                inputs = inputs[:allowed_max_length]
                targets = targets[:allowed_max_length]
        
            inputs_list.append(inputs)
            targets_list.append(targets)
    
        inputs_tensor = torch.stack(inputs_list, dim = 0).to(device)
        targets_tensor = torch.stack(targets_list, dim = 0).to(device)
        return inputs_tensor, targets_tensor

        
    

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        
        for entry in data:
            input_text = self.format_input(entry)
            response_text = f"\n\n### Response: \n {entry['output']}"
            full_text = input_text + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))
            
    def __getitem__(self, index):
        return self.encoded_texts[index]
    
    
    def __len__(self):
        return len(self.data)
    
    
    def format_input(self, entry):
        instruction_text = (
            f"Below is an instruction that describes the task. "
            f"Write a response that appropriately complete the request. "
            f"\n\n### Instruction: \n{entry['instruction']}" )
        input_text = (f"\n\n### Input: \n{entry['input']}" if entry['input'] else "")
        return instruction_text + input_text
    

    

if __name__ == "__main__": 
    file_path = "/Users/santosh/Documents/workspace/data/ml/dialogue.json"
    pd = PrepareDataset(file_path, None, 2, 0)
    
    batch = ([1, 2, 3, 4, 5], [6, 7], [9, 10, 11])
    inputs, targets = pd.collate_fn(batch)
    print(inputs)
    print(targets)

            
        
        