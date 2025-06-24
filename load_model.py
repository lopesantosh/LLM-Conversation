import os
import json
import numpy as np
import tensorflow as tf


def load_gpt2(model_dir):
    
    # load settings and params
    ckpt_path = tf.train.latest_checkpoint(model_dir)

    # load settings
    file_path = os.path.join(model_dir, "hparams.json")
    file = open(file_path, "r", encoding = "utf-8")
    settings = json.load(file)

    # load parameters
    params = load_gpt2_params(ckpt_path, settings)
    return settings, params


def load_gpt2_params(ckpt_path, settings):
    
    # initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        
        #load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))
        

        # process the variable name to extract relevant parts
        parts = name.split("/")[1:]  # skip the 'model/' prefix

        # identify the target dictionary for the variable
        target_dict = params
        if parts[0].startswith("h"):
            layer_number = int(parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # recursively access or create nested dictionaries
        for key in parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # assign the variable array to the last key
        last_key = parts[-1]
        target_dict[last_key] = variable_array

    return params

if __name__ == "__main__":
    model_dir = "/Users/santosh/Documents/workspace/models/gpt2/medium"
    settings, params = load_gpt2(model_dir)