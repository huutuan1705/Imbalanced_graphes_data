import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.mini_imagenet import MiniImageNet, TaskGenerator
from models.maml import ConvNet, MAML
from utils.utils import set_seed, visualize_samples


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MAML model on Few-Shot Learning tasks')
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, default='./data', 
                        help='Path to the dataset')
    
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
    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the pretrained model')
    
    # Training arguments
    parser.add_argument('--inner_lr', type=float, default=0.01, 
                        help='Inner loop learning rate')
    parser.add_argument('--num_inner_steps', type=int, default=5, 
                        help='Number of inner loop updates')
    parser.add_argument('--adapt_steps', type=int, nargs='+', default=[1, 3, 5, 10], 
                        help='Number of adaptation steps to evaluate')
    
    # Evaluation arguments
    parser.add_argument('--num_tasks', type=int, default=600, 
                        help='Number of test tasks')
    parser.add_argument('--visualize', action='store_true', 
                        help='Visualize some examples')
    parser.add_argument('--save_dir', type=str, default='./results', 
                        help='Directory to save results')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def evaluate(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Make sure save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Create datasets
    transform = torch.torchvision.transforms.Compose([
        torch.torchvision.transforms.Resize(84),
        torch.torchvision.transforms.CenterCrop(84),
        torch.torchvision.transforms.ToTensor(),
        torch.torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    print(f"Loading test dataset from {args.data_path}...")
    test_dataset = MiniImageNet(args.data_path, mode='test', transform=transform)
    
    # Create task generator
    test_task_generator = TaskGenerator(
        test_dataset, 
        n_way=args.n_way, 
        k_shot=args.k_shot, 
        q_query=args.q_query
    )
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    device = torch.device(args.device)
    model = ConvNet(in_channels=3, num_filters=args.num_filters, num_classes=args.n_way)
    
    # Load trained model
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create MAML model
    maml = MAML(
        model=model,
        inner_lr=args.inner_lr,
        meta_lr=0.001,  # Not used for evaluation
        num_inner_steps=args.num_inner_steps
    )
    maml.to(device)
    
    # Evaluate with different adaptation steps
    results = {}
    
    for num_steps in args.adapt_steps:
        print(f"Evaluating with {num_steps} adaptation steps...")
        maml.num_inner_steps = num_steps
        
        test_losses = []
        test_accs = []
        
        for i in tqdm(range(args.num_tasks)):
            # Generate task
            support_images, support_labels, query_images, query_labels = test_task_generator.generate_batch(1)
            
            # Move to device
            support_images = support_images.to(device)
            support_labels = support_labels.to(device)
            query_images = query_images.to(device)
            query_labels = query_labels.to(device)
            
            # Visualize some examples
            if args.visualize and i == 0:
                visualize_samples(
                    support_images[0], support_labels[0], 
                    args.n_way, args.k_shot, 
                    title=f"{args.n_way}-way {args.k_shot}-shot Support Set"
                )
                visualize_samples(
                    query_images[0][:args.n_way*5], query_labels[0][:args.n_way*5], 
                    args.n_way, 5, 
                    title=f"{args.n_way}-way 5-Query Examples"
                )
            
            # Meta-evaluate
            loss, acc = maml.meta_evaluate(support_images, support_labels, query_images, query_labels)
            test_losses.append(loss)
            test_accs.append(acc)
        
        # Calculate statistics
        test_loss = np.mean(test_losses)
        test_acc = np.mean(test_accs)
        test_acc_ci95 = 1.96 * np.std(test_accs) / np.sqrt(len(test_accs))
        
        results[num_steps] = {
            'loss': test_loss,
            'accuracy': test_acc,
            'ci95': test_acc_ci95
        }
        
        print(f"{num_steps} adaptation steps - Accuracy: {test_acc:.4f} ± {test_acc_ci95:.4f}")
    
    # Save results
    result_file = os.path.join(args.save_dir, f"{args.n_way}way_{args.k_shot}shot_results.txt")
    with open(result_file, 'w') as f:
        f.write(f"Evaluation Results for {args.n_way}-way {args.k_shot}-shot:\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Inner Learning Rate: {args.inner_lr}\n")
        f.write("=" * 50 + "\n\n")
        
        for steps, metrics in results.items():
            f.write(f"Adaptation Steps: {steps}\n")
            f.write(f"  Loss: {metrics['loss']:.4f}\n")
            f.write(f"  Accuracy: {metrics['accuracy']:.4f} ± {metrics['ci95']:.4f}\n\n")
    
    # Plot results
    steps = list(results.keys())
    accuracies = [results[s]['accuracy'] for s in steps]
    errors = [results[s]['ci95'] for s in steps]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(steps, accuracies, yerr=errors, marker='o', linestyle='-', capsize=5)
    plt.xlabel('Number of Adaptation Steps')
    plt.ylabel('Accuracy')
    plt.title(f'MAML Accuracy for {args.n_way}-way {args.k_shot}-shot with Different Adaptation Steps')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(steps)
    plt.ylim(bottom=max(0, min(accuracies) - 0.1), top=min(1.0, max(accuracies) + 0.1))
    
    plot_file = os.path.join(args.save_dir, f"{args.n_way}way_{args.k_shot}shot_adaptation_steps.png")
    plt.savefig(plot_file)
    plt.close()
    
    print(f"Results saved to {result_file}")
    print(f"Plot saved to {plot_file}")
    
    return results


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)