import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.nn.functional import cross_entropy
from torchmetrics import F1Score
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from models.example_model import ExModel
from datasets.dataset_retrieval import custom_dataset
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import tqdm
import sklearn.metrics
import numpy as np

import os

save_model_path = "checkpoints/"
pth_name = "saved_model.pth"




def val(model, data_val, loss_function, writer, epoch, device):
    f1score = 0
    f1 = F1Score(num_classes=2, task = 'binary')
    data_iterator = enumerate(data_val)  # take batches
    f1_list = []
    f1t_list = []

    correct = 0
    total = 0

    # Initialize counters for class-wise accuracy
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    with torch.no_grad():
        model.eval()  # switch model to evaluation mode
        tq = tqdm.tqdm(total=len(data_val))
        tq.set_description('Validation:')

        total_loss = 0

        for _, batch in data_iterator:
            # forward propagation
            image, label = batch
            image = image.to(device)
            label = label.to(device)
            pred = model(image)

            loss = loss_function(pred, label.float())

            pred = pred.softmax(dim=1)
            
            f1_list.extend(torch.argmax(pred, dim =1).tolist())
            f1t_list.extend(torch.argmax(label, dim =1).tolist())

            # Accuracy
            _, predicted = torch.max(pred, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

            # Calculate class-wise accuracy
            for i in range(len(label)):
                if torch.argmax(label[i]) == predicted[i]:
                    class_correct[torch.argmax(label[i]).item()] += 1
                class_total[torch.argmax(label[i]).item()] += 1

            total_loss += loss.item()
            tq.update(1)
            

    accuracy = correct / total
    f1score = f1(torch.tensor(f1_list), torch.tensor(f1t_list))
    writer.add_scalar("Validation F1", f1score, epoch)
    writer.add_scalar("Validation Loss", total_loss/len(data_val), epoch)
    writer.add_scalar("Validation Accuracy", accuracy, epoch)
    
    # Calculate and log class-wise accuracy
    for i in range(2):
        writer.add_scalar(f"Class {i} Accuracy", class_correct[i] / class_total[i], epoch)

    f1_scores = sklearn.metrics.f1_score(np.array(f1t_list), np.array(f1_list), average=None)
    writer.add_scalar("F1 Score for class 1", f1_scores[0], epoch)
    writer.add_scalar("F1 Score for class 2", f1_scores[1], epoch)

    tq.close()
    print("F1 score: ", f1score)
    

    return None


def train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, device):
    # runs store the logs for tensorboard for each configuration (ResNet18_Adam, VGG16_SGD, etc.)
    writer = SummaryWriter(log_dir="runs/ResNet18_Adam")

    model.to(device)  # Move the model to the specified device (e.g., GPU or CPU)
    model.train()  # Set the model to training mode
    for epoch in range(n_epochs):
        
        
        model.train()
        running_loss = 0.0
        
        tq = tqdm.tqdm(total=len(train_loader))
        tq.set_description('epoch %d' % (epoch))
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)  # Move the batch of images to the specified device
            labels = labels.to(device)  # Move the batch of labels to the specified device
            
            optimizer.zero_grad()  # Reset the gradients of the optimizer
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = loss_fn(outputs, labels)
            outputs = outputs.softmax(dim=1)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()
            
            running_loss += loss.item()
            tq.set_postfix(loss_st='%.6f' % loss.item())
            tq.update(1)
        
        writer.add_scalar("Training Loss", running_loss/len(train_loader), epoch)
           
        tq.close()
        epoch_loss = running_loss / len(train_loader)
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, n_epochs, epoch_loss))
                
        #check the performance of the model on unseen dataset4
        val(model, val_loader, loss_fn, writer, epoch, device)
        
        #save the model in pth format
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(checkpoint, os.path.join(save_model_path, pth_name))
        print("saved the model " + save_model_path)




def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([100, 100]),
        transforms.RandomHorizontalFlip(),   # Flip horizontally
        transforms.RandomRotation(10),       # Rotation by 20 degrees
        transforms.RandomVerticalFlip(),     # Flip vertically
        transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),  # Random affine transformation for zoom
        transforms.ColorJitter(contrast=0.5) 
    ])

    train_data = custom_dataset("train", transforms=tr)
    val_data = custom_dataset("validate", transforms= tr)


    train_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=2,
        drop_last=True
    )

    model = ExModel().to(device)   # Initialsing an object of the class.

    # Learning rate
    # 0.001 for Adam with ResNet18
    # 0.001 for SGD with ResNet18
    # 0.001 for Adam with VGG16
    # 0.01 for SGD with VGG16
    optimizer = Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    #  15 for ResNet18 with Adam or SGD
    #  10 for VGG16 with Adam
    #  12 for VGG16 with SGD
    max_epoch = 15


    train(model, train_loader, val_loader,  optimizer, loss, max_epoch, device)
    
    
if __name__ == "__main__":
    main()