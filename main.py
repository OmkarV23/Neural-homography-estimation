from HomographyNet import Net
from data import data
from torchsummary import summary
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tqdm
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./Tensorboard_summary')

train_data, valid_data, test_data= data()

train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=32, shuffle=False)
# test_dataloader = DataLoader(test_data, batch_size=16, shuffle=False)

model = Net()
model = model.model().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.1)

def save_checkpoint(save_path, model, optimizer, val_loss):
    if save_path==None:
        return
    save_path = save_path 
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'val_loss': val_loss}

    torch.save(state_dict, save_path)

    print(f'Model saved to ==> {save_path}')

def train_net(model, train_loader, valid_loader, num_epochs):
    best_val_loss = float('Inf')
    train_losses=[]
    val_losses=[]
    
    for i in range(num_epochs):
        current_train_loss=0.0
        model.train()
        print("Epoch {%d}"%(i+1))
        for input,gt in tqdm.tqdm(train_loader):
            input = input.to(device)
            gt = gt.to(device)

            writer.add_graph(model, input)

            output = torch.unsqueeze(model(input),1)
            loss = criterion(output, gt)

            #Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_train_loss+=loss.item()

        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], i)

        #Calculate training loss
        average_training_loss = current_train_loss/len(train_data)
        train_losses.append(average_training_loss)

        current_val_loss = 0.0
        with torch.no_grad():

            model.eval()
            for input,gt in tqdm.tqdm(valid_loader):
                input = input.to(device)
                gt = gt.to(device)
                output = model(input)
                loss = criterion(output, gt)
                current_val_loss += loss.item()

        average_validation_loss =  current_val_loss/ len(valid_data)
        val_losses.append(average_validation_loss)

        writer.add_scalar('Loss/train', average_training_loss, i)
        writer.add_scalar('Loss/valid', average_validation_loss, i)

        print('Epoch [{}/{}],Train Loss: {:.4f}, Valid Loss: {:.8f}'
                .format(i+1, num_epochs, average_training_loss, average_validation_loss))
        if average_validation_loss < best_val_loss:
            best_val_loss = average_validation_loss
            save_name = 'checkpoints/best_model_{}.pt'.format(i+1)
            save_checkpoint(save_name, model, optimizer, best_val_loss)
    writer.close()

if __name__ == '__main__':
    train_net(model,train_dataloader,valid_dataloader,100)
