import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from torch.utils.data import DataLoader, TensorDataset
from datahandler import create_input_image, extract_patches_and_labels, process_3d_mask
from tabulate import tabulate 

class GBD_CNN(torch.nn.Module):
    def __init__(self, input_channels=5):
        super(GBD_CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(input_channels, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.dropout1 = torch.nn.Dropout2d(0.5)
        self.dropout2 = torch.nn.Dropout2d(0.5)
        self.fc1 = torch.nn.Linear(8*5*5, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 2)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.dropout1(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.dropout2(x)
        x = x.view(-1, 8*5*5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        return self.fc3(x)

def prepare_test_data(image_path, mask_path):
    rgb_image = plt.imread(image_path)
    height, width = rgb_image.shape[:2]
    
    cutout_width = 763
    cutout_height = 434
    x_start = width // 2
    y_start = 100
    
    cutout = rgb_image[y_start:y_start+cutout_height, x_start:x_start+cutout_width]
    
    data = np.load(mask_path)
    mask_3d = data['arr_0'][y_start:y_start+cutout_height, x_start:x_start+cutout_width, :]
    mask_2d = process_3d_mask(mask_3d)
    
    return cutout, mask_2d

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def visualize_results(original, mask_gt, mask_pred, mode, split):
    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Pôvodný výrez")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(mask_gt, cmap='gray')
    plt.title("Skutočná maska")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred, cmap='gray')
    plt.title(f"Predikcia {mode}-{split}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_channels = {'RGB':3, 'RGBL':4, 'RGBLCon':5, 'HSV':3, 'HSVL':4, 'HSVLCon':5}
    
    rgb_test, mask_test = prepare_test_data("yelow/CDY_2018.jpg", "yelow/CDY_2018.npz")
    
    results_table = []
    models_dir = "modelsyellow"
    for model_file in os.listdir(models_dir):
        if not model_file.endswith(".pth"):
            continue
            
        mode, split = model_file[:-4].split("_")
        print(f"\nVyhodnocujem model: {mode} ({split})")
        
        input_image = create_input_image(rgb_test, mode, is_blue_grape=False)
        
        patches, labels = extract_patches_and_labels(input_image, mask_test)
        
        test_dataset = TensorDataset(
            torch.FloatTensor(patches).permute(0, 3, 1, 2),
            torch.LongTensor(labels)
        )
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        model = GBD_CNN(input_channels=mode_channels[mode]).to(device)
        model.load_state_dict(torch.load(os.path.join(models_dir, model_file)))
        
        preds, true_labels = evaluate_model(model, test_loader, device)
        
        # Výpočet confusion matrix
        TP = np.sum((preds == 1) & (true_labels == 1))
        TN = np.sum((preds == 0) & (true_labels == 0))
        FP = np.sum((preds == 1) & (true_labels == 0))
        FN = np.sum((preds == 0) & (true_labels == 1))
        
        # Výpočet metrík
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0
        PREC = TP / (TP + FP) if (TP + FP) > 0 else 0
        IOU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        F1 = 2 * (PREC * TPR) / (PREC + TPR) if (PREC + TPR) > 0 else 0
        
        print(f"ACC: {ACC:.4f}")
        print(f"TPR: {TPR:.4f}")
        print(f"TNR: {TNR:.4f}")
        print(f"PREC: {PREC:.4f}")
        print(f"IOU: {IOU:.4f}")
        print(f"F1: {F1:.4f}")
        
        mask_pred = np.zeros_like(mask_test)
        padding = 2
        h, w = mask_test.shape
        mask_pred[padding:h-padding, padding:w-padding] = preds.reshape(h-2*padding, w-2*padding)
        
        visualize_results(rgb_test, mask_test, mask_pred, mode, split)

        results_table.append([
            f"{mode}_{split}",
            round(ACC, 4),
            round(TPR, 4),
            round(TNR, 4),
            round(PREC, 4),
            round(IOU, 4),
            round(F1, 4)
        ])
    
    headers = ["Model", "ACC", "TPR", "TNR", "PREC", "IOU", "F1"]
    print("\n=== Súhrnná tabuľka výsledkov ===")
    print(tabulate(results_table, headers=headers, tablefmt="grid"))

if __name__ == "__main__":
    main()