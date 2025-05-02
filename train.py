import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import torchvision.transforms as transforms

from datasets.mini_imagenet import MiniImageNet, TaskGenerator
from models.maml import ConvNet, MAML
from utils.utils import set_seed, Logger, make_nested_directory, visualize_samples


def parse_args():
    parser = argparse.ArgumentParser(description='MAML for Few-Shot Learning')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='./data', 
                        help='Path to the dataset')
    parser.add_argument('--download', action='store_true', 
                        help='Download the dataset if not available')
    
    # Task arguments
    parser.add_argument('--n_way', type=int, default=5, 
                        help='Number of classes per task (N-way)')
    parser.add_argument('--k_shot', type=int, default=1, 
                        help='Number of samples per class (K-shot)')
    parser.add_argument('--q_query', type=int, default=15, 
                        help='Number of query samples per class')
    
    # Model arguments
    parser.add_argument('--num_filters', type=int, default=32, 
                        help='Number of filters in the ConvNet')
    
    # Training arguments
    parser.add_argument('--inner_lr', type=float, default=0.01, 
                        help='Inner loop learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001, 
                        help='Meta learning rate')
    parser.add_argument('--num_inner_steps', type=int, default=5, 
                        help='Number of inner loop updates')
    parser.add_argument('--meta_batch_size', type=int, default=4, 
                        help='Number of tasks per batch')
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--eval_interval', type=int, default=1, 
                        help='Evaluation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=10, 
                        help='Model save interval in epochs')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--log_dir', type=str, default='./logs', 
                        help='Directory to save logs and models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    
    return parser.parse_args()


def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.n_way}way_{args.k_shot}shot_{timestamp}")
    log_dir = make_nested_directory(log_dir)
    logger = Logger(log_dir)
    
    # Write configuration to log
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Create datasets
    transform = transforms.Compose([  
        transforms.Resize(84),        
        transforms.CenterCrop(84),    
        transforms.ToTensor(),        
        transforms.Normalize(         
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"Loading datasets from {args.data_path} with download={args.download}...")
    download_dataset = False
    if args.download:
        print("Downloading datasets...")
        download_dataset = True
    elif not os.path.exists(args.data_path):
        print(f"Dataset path {args.data_path} does not exist. Please download the dataset.")
        download_dataset = True
    
    if download_dataset:
        train_dataset = MiniImageNet(args.data_path, mode='train', transform=transform, download=True)
        val_dataset = MiniImageNet(args.data_path, mode='val', transform=transform, download=True)
        test_dataset = MiniImageNet(args.data_path, mode='test', transform=transform, download=True)
    else:
        train_dataset = MiniImageNet(args.data_path, mode='train', transform=transform)
        val_dataset = MiniImageNet(args.data_path, mode='val', transform=transform)
        test_dataset = MiniImageNet(args.data_path, mode='test', transform=transform)
    
    # Create task generators
    train_task_generator = TaskGenerator(
        train_dataset, 
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        q_query=args.q_query
    )
    
    val_task_generator = TaskGenerator(
        val_dataset, 
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        q_query=args.q_query
    )
    
    test_task_generator = TaskGenerator(
        test_dataset, 
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        q_query=args.q_query
    )
    
    # Create model
    print(f"Creating model...")
    device = torch.device(args.device)
    model = ConvNet(in_channels=3, num_filters=args.num_filters, num_classes=args.n_way)
    model.initialize_weights()
    
    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        num_inner_steps=args.num_inner_steps
    )
    maml.to(device)
    
    # Training loop
    print(f"Starting training with {args.n_way}-way {args.k_shot}-shot learning...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        maml.train()
        train_losses = []
        train_accs = []
        
        # Generate a batch of tasks for this epoch
        for i in range(0, 100, args.meta_batch_size):  # Train on 100 tasks per epoch
            # Sample a batch of tasks
            support_images, support_labels, query_images, query_labels = train_task_generator.generate_batch(args.meta_batch_size)
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Meta-train on this batch
            loss, acc = maml.meta_train(support_images, support_labels, query_images, query_labels)
            train_losses.append(loss)
            train_accs.append(acc)
        
        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        
        # Validation
        if epoch % args.eval_interval == 0:
            maml.eval()
            val_losses = []
            val_accs = []
            
            # Evaluate on 100 tasks
            for i in range(100):
                support_images, support_labels, query_images, query_labels = val_task_generator.generate_batch(1)
                
                # Move to device
                support_images = support_images.to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.to(device)
                query_labels = query_labels.to(device)
                
                # Meta-evaluate on this task
                loss, acc = maml.meta_evaluate(support_images, support_labels, query_images, query_labels)
                val_losses.append(loss)
                val_accs.append(acc)
            
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            
            # Log metrics
            logger.log(epoch, train_loss, train_acc, val_loss, val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                logger.save_model(maml.model, 'best_model.pth')
                
                print(f"New best validation accuracy: {val_acc:.4f}")
        else:
            # Log only training metrics
            logger.log(epoch, train_loss, train_acc)
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            logger.save_model(maml.model, f'model_epoch{epoch}.pth')
    
    # Final testing
    print(f"Training completed. Running final evaluation on test set...")
    maml.eval()
    test_losses = []
    test_accs = []
    
    # Test on 600 tasks as in the paper
    for i in tqdm(range(600)):
        support_images, support_labels, query_images, query_labels = test_task_generator.generate_batch(1)
        
        # Move to device
        support_images = support_images.to(device)
        support_labels = support_labels.to(device)
        query_images = query_images.to(device)
        query_labels = query_labels.to(device)
        
        # Meta-evaluate on this task
        loss, acc = maml.meta_evaluate(support_images, support_labels, query_images, query_labels)
        test_losses.append(loss)
        test_accs.append(acc)
    
    test_loss = np.mean(test_losses)
    test_acc = np.mean(test_accs)
    test_acc_ci95 = 1.96 * np.std(test_accs) / np.sqrt(len(test_accs))
    
    print(f"Final test accuracy: {test_acc:.4f} ± {test_acc_ci95:.4f}")
    
    # Log final test results
    with open(os.path.join(log_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f} ± {test_acc_ci95:.4f}\n")
    
    # Plot training curves
    logger.plot_metrics()
    
    # Save training history
    logger.save_history()
    
    print(f"All results saved to {log_dir}")
    return test_acc


if __name__ == '__main__':
    args = parse_args()
    train(args)