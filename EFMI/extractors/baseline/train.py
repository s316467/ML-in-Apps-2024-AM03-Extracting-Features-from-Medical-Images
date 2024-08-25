import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils.plotting import *


# Example of how to use the model
def fine_tune(model, train_loader, num_epochs, model_name):
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        print("Computing loss for epoch: ", epoch + 1)
        running_loss = 0.0
        for inputs, labels, _, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss):
                print(f"NaN loss encountered at epoch {epoch+1}")
                continue
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')
       
        torch.save(
            model.state_dict(),
            f"./results/baselines/finetuned/{model_name}_{epoch}.pth",
          )
    
    return model

def train(model, dataloader, num_epochs, model_name):
    current_epochs = 0
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    training_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (inputs, labels, _, _) in enumerate(
            tqdm(dataloader, desc=f"Epoch {epoch+current_epochs+1}/{num_epochs}")
        ):
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(
                non_blocking=True
            )
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print(f"NaN loss encountered at epoch {epoch+1}")
                    continue
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        epoch_loss = running_loss / len(dataloader)
        training_losses.append(epoch_loss)
        print(f"Epoch {epoch+current_epochs+1}/{num_epochs}, Loss: {epoch_loss}")
        

    # Save the trained model
    torch.save(
        model.state_dict(),
        f"./results/baseline/finetuned/{model_name}_{num_epochs}epochs.pth",
    )
    # Plot and save the training loss
    plot_training_loss(
        training_losses,
        "./results/baseline/finetuned/",
        f"{model_name}_{num_epochs}_trainingloss.png",
    )

    return model
