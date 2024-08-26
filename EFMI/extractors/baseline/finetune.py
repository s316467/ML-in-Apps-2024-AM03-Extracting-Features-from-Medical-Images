import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.plotting import *


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