import torch
import torch.nn as nn


def loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # logits.shape (batch_size, seq_len, vocab_size)
    loss = nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss



def loss_loader(data_loader, model, device, num_batches = None):
    """ Return average loss over number of batches. """
    
    total_loss = 0

    # iterate over all batches if no fixed num_batches is specified
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = loss_batch(input_batch, target_batch, model, device)
            total_loss += loss
        else:
            break
    return total_loss/num_batches
    

    
    
    