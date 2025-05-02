import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """Simple logger to track and save training metrics"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize history dictionary
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        self.log_file = os.path.join(log_dir, 'training_log.txt')
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
        
    def log(self, epoch, train_loss, train_acc, val_loss=None, val_acc=None, test_loss=None, test_acc=None):
        """Log metrics for current epoch"""
        # Update history
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
        if test_loss is not None:
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
        
        # Print to console
        log_message = f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
        if val_loss is not None:
            log_message += f", Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        if test_loss is not None:
            log_message += f", Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
        
        print(log_message)
        
        # Write to log file
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def save_model(self, model, filename):
        """Save model checkpoint"""
        save_path = os.path.join(self.log_dir, filename)
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        # Add to log file
        with open(self.log_file, 'a') as f:
            f.write(f"Model saved to {filename} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
    def plot_metrics(self, save_fig=True):
        """Plot training and validation metrics"""
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        if self.history['val_loss']:
            plt.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        if self.history['test_loss']:
            plt.plot(epochs, self.history['test_loss'], 'g-', label='Test Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        if self.history['val_acc']:
            plt.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        if self.history['test_acc']:
            plt.plot(epochs, self.history['test_acc'], 'g-', label='Test Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.log_dir, 'metrics.png'))
            print(f"Metrics plot saved to {os.path.join(self.log_dir, 'metrics.png')}")
        
        plt.show()
        
    def save_history(self):
        """Save training history to a file"""
        import json
        with open(os.path.join(self.log_dir, 'history.json'), 'w') as f:
            # Convert numpy values to Python scalars
            clean_history = {}
            for key, values in self.history.items():
                clean_history[key] = [float(v) for v in values]
            json.dump(clean_history, f)


def visualize_samples(images, labels, n_way, k_shot, title=None):
    """
    Visualize some samples for N-way K-shot classification
    
    Args:
        images: Tensor of shape [N*K, C, H, W]
        labels: Tensor of shape [N*K]
        n_way: Number of classes
        k_shot: Number of samples per class
        title: Optional title for the plot
    """
    # Assuming images are normalized, denormalize them
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    plt.figure(figsize=(10, 2 * n_way))
    
    for i in range(n_way):
        for j in range(k_shot):
            idx = i * k_shot + j
            plt.subplot(n_way, k_shot, idx + 1)
            
            # Convert tensor to numpy and transpose from (C, H, W) to (H, W, C)
            img = images_denorm[idx].permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.title(f"Class {labels[idx].item()}")
            plt.axis('off')
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def make_nested_directory(path):
    """Create nested directory structure if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path