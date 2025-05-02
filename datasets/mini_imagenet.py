import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import zipfile
import shutil
import random
import glob
from tqdm import tqdm

class MiniImageNet(Dataset):
    """
    Mini-ImageNet dataset for N-way K-shot learning tasks
    """
    def __init__(self, root, mode='train', transform=None, download=False):
        """
        Args:
            root (string): Directory containing the Mini-ImageNet dataset
            mode (string): 'train', 'val', or 'test'
            transform (callable, optional): Transform to be applied on images
            download (bool, optional): If True, downloads the dataset
        """
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.mode = mode
        
        if download:
            self._download()
            
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use download=True to download it')
            
        # Load data based on mode (train/val/test)
        self.data_folder = os.path.join(self.root, 'mini-imagenet', mode)
        self.class_folders = [os.path.join(self.data_folder, class_name) 
                             for class_name in os.listdir(self.data_folder)
                             if os.path.isdir(os.path.join(self.data_folder, class_name))]
        
        self.all_images = []
        self.all_labels = []
        
        for i, class_folder in enumerate(self.class_folders):
            images = [os.path.join(class_folder, img) for img in os.listdir(class_folder) 
                      if img.endswith('.jpg') or img.endswith('.png') or img.endswith('.JPEG')]
            self.all_images.extend(images)
            self.all_labels.extend([i] * len(images))
            
        self.classes = sorted(os.listdir(self.data_folder))
        
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        label = self.all_labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'mini-imagenet', 'train')) and \
               os.path.exists(os.path.join(self.root, 'mini-imagenet', 'val')) and \
               os.path.exists(os.path.join(self.root, 'mini-imagenet', 'test'))
    
    def _download(self):
        """
        Download and process the Mini-ImageNet dataset
        """
        if self._check_exists():
            print("Dataset already exists. Skipping download.")
            return
            
        download_dir = os.path.join(self.root, 'download')
        os.makedirs(download_dir, exist_ok=True)
        
        zip_path = os.path.join(download_dir, 'miniimagenet.zip')
        extracted_dir = os.path.join(download_dir, 'extracted')
        target_dir = os.path.join(self.root, 'mini-imagenet')
        
        if not os.path.exists(zip_path):
            print("Downloading mini-ImageNet dataset...")
            try:
                cmd = f"wget -O {zip_path} 'https://www.kaggle.com/api/v1/datasets/download/arjunashok33/miniimagenet'"
                subprocess.run(cmd, shell=True, check=True)
                print("Download completed.")
            except:
                print("Failed to download using wget. Please download manually from:")
                print("https://www.kaggle.com/datasets/arjunashok33/miniimagenet")
                print(f"Save the file to: {zip_path}")
                return
        
        # Extract dataset if not already extracted
        if not os.path.exists(extracted_dir):
            print("Extracting dataset...")
            os.makedirs(extracted_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extracted_dir)
            print("Extraction completed.")
        
        # Create train, val, test splits
        print("Preparing train/val/test splits...")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'val'), exist_ok=True)
        os.makedirs(os.path.join(target_dir, 'test'), exist_ok=True)
        
        # Get all class folders
        class_folders = [f for f in os.listdir(extracted_dir) 
                        if os.path.isdir(os.path.join(extracted_dir, f))]
        
        for class_folder in tqdm(class_folders, desc="Processing classes"):
            src_folder = os.path.join(extracted_dir, class_folder)
            
            # Get all images in this class
            all_images = glob.glob(os.path.join(src_folder, '*.JPEG'))
            
            # Shuffle images to ensure random split
            random.shuffle(all_images)
            
            # Per plan: Use 60 images for train, 20 for val, 20 for test
            train_images = all_images[:60]
            val_images = all_images[60:80]
            test_images = all_images[80:100]
            
            # Create class folders in train, val, test
            os.makedirs(os.path.join(target_dir, 'train', class_folder), exist_ok=True)
            os.makedirs(os.path.join(target_dir, 'val', class_folder), exist_ok=True)
            os.makedirs(os.path.join(target_dir, 'test', class_folder), exist_ok=True)
            
            # Copy images to respective folders
            for img in train_images:
                shutil.copy(img, os.path.join(target_dir, 'train', class_folder))
            
            for img in val_images:
                shutil.copy(img, os.path.join(target_dir, 'val', class_folder))
                
            for img in test_images:
                shutil.copy(img, os.path.join(target_dir, 'test', class_folder))
        
        print("Dataset preparation completed.")
        print(f"Train images: ~{len(class_folders) * 60}")
        print(f"Val images: ~{len(class_folders) * 20}")
        print(f"Test images: ~{len(class_folders) * 20}")

def get_mini_imagenet_dataloaders(root, batch_size, num_workers=4, download=False):
    """
    Returns train, validation and test dataloaders for Mini-ImageNet
    """
    transform = transforms.Compose([
        transforms.Resize(84),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = MiniImageNet(root, mode='train', transform=transform, download=download)
    val_dataset = MiniImageNet(root, mode='val', transform=transform, download=download)
    test_dataset = MiniImageNet(root, mode='test', transform=transform, download=download)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


class TaskGenerator:
    """
    Class for generating N-way K-shot tasks from Mini-ImageNet
    """
    def __init__(self, dataset, n_way=5, k_shot=1, q_query=15):
        """
        Args:
            dataset: Mini-ImageNet dataset
            n_way: Number of classes in a task
            k_shot: Number of support samples per class
            q_query: Number of query samples per class
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        # Group images by class
        self.class_images = {}
        for idx, (_, label) in enumerate(dataset):
            if label not in self.class_images:
                self.class_images[label] = []
            self.class_images[label].append(idx)
    
    def generate_task(self):
        """
        Generates a single N-way K-shot task
        
        Returns:
            support_images: Support set images [N*K, C, H, W]
            support_labels: Support set labels [N*K]
            query_images: Query set images [N*Q, C, H, W]
            query_labels: Query set labels [N*Q]
        """
        # Randomly sample N classes
        selected_classes = np.random.choice(list(self.class_images.keys()), 
                                          size=self.n_way, replace=False)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        # For each selected class
        for i, cls in enumerate(selected_classes):
            # Get all images of this class
            cls_img_idxs = self.class_images[cls]
            
            # Randomly select K support and Q query images
            selected_imgs = np.random.choice(cls_img_idxs, 
                                           size=self.k_shot + self.q_query,
                                           replace=False)
            support_idxs = selected_imgs[:self.k_shot]
            query_idxs = selected_imgs[self.k_shot:self.k_shot + self.q_query]
            
            # Add to support and query sets with new labels (0 to N-1)
            for idx in support_idxs:
                img, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(i)
                
            for idx in query_idxs:
                img, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(i)
        
        # Stack all images and convert labels to tensors
        support_images = torch.stack(support_images)
        support_labels = torch.tensor(support_labels)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels)
        
        return support_images, support_labels, query_images, query_labels
    
    def generate_batch(self, batch_size):
        """
        Generates a batch of N-way K-shot tasks
        
        Args:
            batch_size: Number of tasks in the batch
            
        Returns:
            batch of tasks
        """
        support_images_list = []
        support_labels_list = []
        query_images_list = []
        query_labels_list = []
        
        for _ in range(batch_size):
            support_images, support_labels, query_images, query_labels = self.generate_task()
            support_images_list.append(support_images)
            support_labels_list.append(support_labels)
            query_images_list.append(query_images)
            query_labels_list.append(query_labels)
        
        return (torch.stack(support_images_list), torch.stack(support_labels_list),
                torch.stack(query_images_list), torch.stack(query_labels_list))