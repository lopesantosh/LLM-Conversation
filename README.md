# Finetune a Large language Model for Text Conversation Between Two People

This repository contains the code for developing and finetuning a GPT-like Large Language Model (LLM) for text conversation between two people. 

# Requirements
All the required python packages could be installed using pip install -r requirements.txt

# Running Python file
For finetuning, run the python file finetuning.py. It read the input dataset from the current folder, load and download (create a local folder if running for the first time) the language model GPT2, and run the training loop. Afer finetuning the LLM model, the script outputs a file in JSON format in a current folder which contains the result of the generated conversation, along with the true conversation from the validation set (three cases).

# Hardware Requirements
The code is designed to run on conventional laptops within a reasonable timeframe and does not require specialized hardware. For the given data set here, it takes less than 4-5 minutes to run. Additionally, the code automatically utilizes GPUs if they are available.
