import torch

from prepare_data import PrepareDataset
from gptmodel import get_model
from loss import loss_loader
from train import train_model

def main(data_path, model_dir):
    torch.manual_seed(123)
    
    ##########################################
    # set up configuration and prepare dataset
    ##########################################
    num_workers = 0
    batch_size = 2
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    file_path = data_path
    pd = PrepareDataset(file_path, None, batch_size, num_workers)
    train_loader = pd.train_loader
    val_loader = pd.val_loader


    ##########################################
    # load pretrained model
    ##########################################
    model_name = "gpt2-medium"
    base_dir = model_dir
    model = get_model(model_name, base_dir)
    model.eval()
    model.to(device)

    print("Loaded model:", model_name)
    print(50*"-")

    #######################################
    # Finetuning the model
    #######################################
    print("Initial losses")
    with torch.no_grad():
        train_loss = loss_loader(train_loader, model, device, num_batches=4)
        val_loss = loss_loader(val_loader, model, device, num_batches=4)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    # use adamw optimizer, learning weight and weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    # training loops
    train_model(model, train_loader, val_loader, optimizer, device, num_epochs, 2, 2 )
                                                              
    print(50*"-")

    #######################################
    # Evaluating and Saving results
    #######################################
    print("Generating conversational responses")
    
        


    
    

if __name__ == "__main__":
    file_path = "/Users/santosh/Documents/workspace/data/ml/dialogue.json"
    model_path = "/Users/santosh/Documents/workspace/models"
    main(file_path, model_path)