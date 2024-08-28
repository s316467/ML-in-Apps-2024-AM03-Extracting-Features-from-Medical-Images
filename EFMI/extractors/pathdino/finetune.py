import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from utils.plotting import *


def fine_tune(model, dataloader, num_epochs):
    current_epochs = 0

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scaler = GradScaler()  # Mixed precision training scaler

    criterion = nn.CrossEntropyLoss()
    model = model.cuda()

    for param in model.parameters():
        param.requires_grad = False

    # Last ViT block only
    for param in model.blocks[-1].parameters():
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
        if (epoch + 1) % 2 == 0 and epoch > 0:
            torch.save(
                model.state_dict(),
                f"./results/pathdino/finetune/pathdino_{epoch+current_epochs+1}.pth",
            )

    # Save the trained model
    torch.save(
        model.state_dict(),
        f"./results/pathdino/finetune/pathdino_{num_epochs}epochs.pth",
    )
    # Plot and save the training loss
    plot_training_loss(
        training_losses,
        "./results/pathdino/finetune/",
        f"pathdino_{num_epochs}_trainingloss.png",
    )

    return model
