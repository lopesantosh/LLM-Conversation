# Finetune a Large language Model for Text Conversation Between Two People

This repository contains the code for developing and finetuning a GPT-like Large Language Model (LLM) for text conversation between two people. 

# Conversation Data Preparation
The datasets contain 10 entries, each presenting a conversation dialogue between two people. I have formatted each entry in a prompt-style format.

To create a batching process for training, It implements a custom collate function to handle the specific requirements and formatting of the converstation dataset. At a high level, the custom collate function performs the following actions:

1. It formats each data entry using a prompt-style format.
2. It tokenizes the formatted data.
3. It pads the training examples within each batch to the same length, while allowing different batches to have different lengths. This approach minimizes unnecessary padding by extending sequences to match the longest one in each batch, rather than the entire dataset.
4. It creates batches with target token IDs that correspond to the batch of input IDs. These target IDs are crucial because they represent what we want the model to generate. The target token IDs match the input token IDs but are shifted one position forward.
5. It assigns a placeholder value of -100 to all padded tokens to exclude them from contributing to the training cross-entropy calculation. This ensures that only meaningful data influences the model training.

I divided the dialogue dataset into training and validation sets, using a 70-30 split. Subsequently, I implemented a DialogueDataset class and applied data loaders to both the training and validation sets, which are necessary for large language model (LLM) fine-tuning and evaluation.

# Model
I chose the GPT-2 (355 million parameters) model due to its relatively simple architecture compared to other transformer architectures. As a decoder-style autoregressive model, it incorporates its previous outputs as inputs to predict the next word sequentially, making it ideal for text generation and conversational tasks. 

For model, one need to specify the choice of the model, dierctory's path to the model. 

model_name - "gpt2-medium" or "gpt2-large" or "gpt2-xl"

Model path - "model_path" where the speciied model would be locally stored. 


# Requirements
All the required python packages could be installed using pip install -r requirements.txt

# Running Python file
For finetuning, run the python file finetune.py. It read the input dataset from the current folder, load and download (create a local folder if running for the first time) the language model GPT2, and run the training loop. Afer finetuning the LLM model, the script outputs a file in JSON format in a current folder which contains the result of the generated conversation, along with the true conversation from the validation set (three cases).

# Hardware Requirements
The code is designed to run on conventional laptops within a reasonable timeframe and does not require specialized hardware. For the given data set here, it takes less than 4-5 minutes to run. Additionally, the code automatically utilizes GPUs if they are available.
