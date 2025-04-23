import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from datahandler import prepare_data, create_input_image, extract_patches_and_labels, get_dataloaders
import os
import random
import matplotlib.pyplot as plt

random.seed(9)
torch.manual_seed(9)

class GBD_CNN(nn.Module):
    def __init__(self, input_channels=5):
        super(GBD_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(8*5*5, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = self.dropout1(x)
        x = nn.ReLU()(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(-1, 8*5*5)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        return self.fc3(x)

def train_model(train_loader, val_loader, input_channels, device, epochs=5):
    model = GBD_CNN(input_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(all_labels, all_preds)
        train_accs.append(train_acc)
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    return model, {
        'train_acc': np.mean(train_accs[-3:]),
        'val_acc': np.mean(val_accs[-3:]),
    }


def run_experiments(image_path, mask_path, modes, is_blue_grape, epochs, x_start, y_start, cutout_size):
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    
    if(is_blue_grape == False):
        rgb_image, mask_2d = prepare_data(image_path, mask_path,x_start,y_start,cutout_size)
    else:
        rgb_image, mask_2d = prepare_data(image_path, mask_path,x_start,y_start,cutout_size)
    plt.figure()
    plt.imshow(rgb_image)
    plt.show()
    
    for mode in modes:
        print(f"\nExperiment for {mode}")
        
        input_image = create_input_image(rgb_image, mode, is_blue_grape)
        patches, labels = extract_patches_and_labels(input_image, mask_2d)
        

        for split_name, split_seed in [('P1', 1), ('P2', 2)]:
            print(f"Running {split_name} with seed {split_seed}")

            train_loader, val_loader = get_dataloaders(
                patches, labels, 
                batch_size=256,
                split_seed=split_seed
            )
            
            result_key = f"{mode}_{split_name}"
            
            model, metrics = train_model(  
                train_loader, val_loader, 
                input_image.shape[2], device, epochs
            )
            results[result_key] = metrics
            
            if is_blue_grape == False:
                model_path = f"modelsyellow/{mode}_{split_name}.pth"
                os.makedirs("modelsyellow", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"Model uložený do: {model_path}")
            else:
                model_path = f"modelsblue/{mode}_{split_name}.pth"
                os.makedirs("modelsblue", exist_ok=True)
                torch.save(model.state_dict(), model_path)
                print(f"Model uložený do: {model_path}")
    
    print("\n=== Final Results ===")
    print("Mode\t\tSplit\tTrain Acc\tVal Acc")
    for key in results:
        mode, split = key.split('_')
        res = results[key]
        print(f"{mode}\t\t{split}\t{res['train_acc']:.4f}\t\t{res['val_acc']:.4f}")

if __name__ == "__main__":
    # pokial chcete menit medzi modreov a zltov meni sa to timto boleanom :D
    blue = True   
    if(blue == True):
        run_experiments(
            image_path="blue/CSV_1898.jpg",
            mask_path="blue/CSV_1898.npz",
            modes=['RGB', 'RGBL', 'RGBLCon', 'HSV', 'HSVL', 'HSVLCon'],
            is_blue_grape=True,
            epochs=5,
            x_start = 800,
            y_start = 400,
            cutout_size = (763,434)
        )
    else:
        run_experiments(
            image_path="yelow/CDY_2015.jpg",
            mask_path="yelow/CDY_2015.npz",
            modes=['RGB', 'RGBL', 'RGBLCon', 'HSV', 'HSVL', 'HSVLCon'],
            is_blue_grape=False,
            epochs=5,
            x_start = 0,
            y_start = 463,
            cutout_size = (763,434)
        )