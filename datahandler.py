import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader  
import random

random.seed(9)
torch.manual_seed(9)

def rgb_to_cmyk(img):
    bgr = img[..., ::-1].astype(np.float32) / 255.0
    eps = 1e-6
    K = 1 - np.max(bgr, axis=2)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        C = (1 - bgr[..., 2] - K) / np.maximum(1 - K, eps)
        M = (1 - bgr[..., 1] - K) / np.maximum(1 - K, eps)
        Y = (1 - bgr[..., 0] - K) / np.maximum(1 - K, eps)
    

    cmyk = np.stack((C, M, Y, K), axis=2)
    return (cmyk * 255).clip(0, 255).astype(np.uint8)

def compute_blue_features(CMYK):
    C = CMYK[..., 0].astype(np.float32) / 255.0
    Y = CMYK[..., 2].astype(np.float32) / 255.0
    L = Y
    Con = (C - Y) / np.maximum(Y, 1.0)
    
    return L, Con

def compute_yellow_features(CMYK):
    C = CMYK[..., 0].astype(np.float32) / 255.0
    M = CMYK[..., 1].astype(np.float32) / 255.0
    
    L = C
    Con = (M - C) / np.maximum(C, 1.0)
    
    return L, Con

def create_input_image(rgb_image, mode, is_blue_grape):
    cmyk_image = rgb_to_cmyk(rgb_image)
    L, Con = compute_blue_features(cmyk_image) if is_blue_grape else compute_yellow_features(cmyk_image)
    
    # print(f"Kontrast - min: {Con.min():.2f}, max: {Con.max():.2f}, mean: {Con.mean():.2f}, std: {Con.std():.2f}")

    rgb_normalized = rgb_image / 255.0
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV) / 255.0
    
    modes = {
        'RGB': rgb_normalized,
        'RGBL': np.dstack((rgb_normalized, L)),
        'RGBLCon': np.dstack((rgb_normalized, L, Con)),
        'HSV': hsv_image,
        'HSVL': np.dstack((hsv_image, L)),
        'HSVLCon': np.dstack((hsv_image, L, Con))
    }
    return modes[mode]

def process_3d_mask(mask_3d):
    return np.any(mask_3d > 0, axis=2).astype(np.int8)

def extract_patches_and_labels(image, mask, window_size=5):
    padding = window_size // 2
    patches, labels = [], []
    
    for y in range(padding, image.shape[0] - padding):
        for x in range(padding, image.shape[1] - padding):
            patch = image[y-padding:y+padding+1, x-padding:x+padding+1, :]
            label = mask[y, x]
            if label in [0, 1]:
                patches.append(patch)
                labels.append(label)
    return np.array(patches), np.array(labels)

def prepare_data(image_path, mask_path, x_start, y_start, cutout_size):

    rgb_image = plt.imread(image_path)

    cutout = rgb_image[y_start:y_start+cutout_size[1], x_start:x_start+cutout_size[0]]
    

    data = np.load(mask_path)
    mask_3d = data['arr_0'][y_start:y_start+cutout_size[1], x_start:x_start+cutout_size[0], :]
    mask_2d = process_3d_mask(mask_3d)
    
    return cutout, mask_2d

def get_dataloaders(patches, labels, batch_size=256, split_seed=1):
    X_trainval, X_test, y_trainval, y_test  = train_test_split(
        patches, labels, 
        test_size=0.1, 
        random_state=split_seed,
        stratify=labels
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.2,
        random_state=split_seed,
        stratify=y_trainval
    )
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train).permute(0, 3, 1, 2),
        torch.LongTensor(y_train)
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val).permute(0, 3, 1, 2),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def visualize_luminance_contrast(rgb_image, is_blue_grape):

    cmyk = rgb_to_cmyk(rgb_image)
    
    if is_blue_grape:
        L, Con = compute_blue_features(cmyk)

    else:
        L, Con = compute_yellow_features(cmyk)

    
    L_norm = cv2.normalize(L, None, 0, 255, cv2.NORM_MINMAX)
    Con_norm = cv2.normalize(Con, None, 0, 255, cv2.NORM_MINMAX)

    plt.figure(figsize=(18, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(rgb_image)

    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(L, cmap='gray')

    plt.axis('off')


    plt.subplot(1, 3, 3)
    plt.imshow(Con, cmap='gray')

    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    blue = True
    if blue == False:
        IMAGE_PATH = "yelow/CDY_2015.jpg"
        MASK_PATH = "yelow/CDY_2015.npz"
        IS_BLUE_GRAPE = False
        

        print("Načítavam dáta...")
        rgb_image = plt.imread(IMAGE_PATH)
        print()
        height, width = rgb_image.shape[:2]
        
        cutout_size=(2000,1300)
        x_start, y_start = 0, 0
        cutout, _ = prepare_data(IMAGE_PATH, MASK_PATH,x_start,y_start,cutout_size)
        
        visualize_luminance_contrast(cutout, IS_BLUE_GRAPE)

    else:
        IMAGE_PATH = "blue/CSV_1898.jpg"
        MASK_PATH = "blue/CSV_1898.npz"
        IS_BLUE_GRAPE = True
        
        rgb_image = plt.imread(IMAGE_PATH)
        height, width = rgb_image.shape[:2]
        
        cutout_size=(2000, 434)
        x_start, y_start = 0,0
        cutout, _ = prepare_data(IMAGE_PATH, MASK_PATH,x_start,y_start,cutout_size)
        

        visualize_luminance_contrast(cutout, IS_BLUE_GRAPE)
