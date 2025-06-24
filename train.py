from loss import loss_batch
from loss import loss_loader

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_batches, 
                start_context = None, tokenizer = None):
    """eval_batches = number of batches used to compute loss in evaluation."""

    train_losses, val_losses = [], []
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                model.eval()
                num_batches = eval_batches # number of batches used to compute loss
                train_loss = loss_loader(train_loader, model, device, num_batches)
                val_loss = loss_loader(val_loader, model, device, num_batches)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                model.train()

                print(f"Epoch: {epoch+1}, Step: {global_step:06d}, "
                      f"Train loss: {train_loss:0.3f}, Val loss: {val_loss:0.3f}")

    return train_losses, val_losses
                
                
            