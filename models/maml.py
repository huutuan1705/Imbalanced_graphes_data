import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class ConvBlock(nn.Module):
    """
    Standard convolutional block used in the MAML paper for mini-ImageNet
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, momentum=1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class ConvNet(nn.Module):
    """
    4-layer Convolutional Network used for mini-ImageNet, as described in the MAML paper
    """
    def __init__(self, in_channels=3, num_filters=32, num_classes=5):
        super(ConvNet, self).__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            ConvBlock(in_channels, num_filters), # 84 -> 42
            ConvBlock(num_filters, num_filters), # 42 -> 21
            ConvBlock(num_filters, num_filters), # 21 -> 10
            ConvBlock(num_filters, num_filters)  # 10 -> 5
        )
        
        # Tính toán kích thước cuối cùng (5x5 cho đầu vào 84x84)
        # Kích thước đầu ra là 5x5xnum_filters
        final_size = 5 * 5 * num_filters
        
        self.classifier = nn.Linear(final_size, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def forward_with_feature(self, x):
        """Forward and return features before the classifier"""
        x = self.features(x)
        feature = x.view(x.size(0), -1)
        logits = self.classifier(feature)
        return logits, feature
    
    def copy_weights(self, model):
        """Copy weights from another model instance"""
        for target_param, param in zip(self.parameters(), model.parameters()):
            target_param.data.copy_(param.data)
    
    def zero_grad(self, set_to_none=False):
        """Overriding zero_grad to handle the case of none gradient"""
        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_()
                    p.grad.zero_()
    
    def initialize_weights(self):
        """Initialize the weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning (MAML) implementation
    """
    def __init__(self, model, inner_lr=0.01, meta_lr=0.001, num_inner_steps=5):
        """
        Args:
            model: Base model to be used (e.g., ConvNet)
            inner_lr: Learning rate for inner/task-specific updates
            meta_lr: Learning rate for meta-update (outer loop)
            num_inner_steps: Number of inner loop updates
        """
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.num_inner_steps = num_inner_steps
        
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.meta_lr)
        
    def forward(self, support_images, support_labels, query_images):
        """
        Performs a forward pass for inference
        
        Args:
            support_images: Support set images [N*K, C, H, W]
            support_labels: Support set labels [N*K]
            query_images: Query set images [N*Q, C, H, W]
            
        Returns:
            query_logits: Logits for query images [N*Q, num_classes]
        """
        device = support_images.device
        
        task_model = ConvNet(in_channels=self.model.in_channels, 
                             num_filters=self.model.num_filters, 
                             num_classes=self.model.num_classes)
        task_model.to(device)
        
        for target_param, param in zip(task_model.parameters(), self.model.parameters()):
            target_param.data.copy_(param.data.to(device))
        
        for _ in range(self.num_inner_steps):
            support_logits = task_model(support_images)
            support_loss = F.cross_entropy(support_logits, support_labels)
            
            grads = torch.autograd.grad(support_loss, task_model.parameters())
            
            for p, g in zip(task_model.parameters(), grads):
                p.data.sub_(self.inner_lr * g)

        query_logits = task_model(query_images)
        return query_logits
    
    def meta_train(self, support_images, support_labels, query_images, query_labels):
        """
        Performs meta-training on a batch of tasks
        
        Args:
            support_images: Support set images [batch_size, N*K, C, H, W]
            support_labels: Support set labels [batch_size, N*K]
            query_images: Query set images [batch_size, N*Q, C, H, W]
            query_labels: Query set labels [batch_size, N*Q]
            
        Returns:
            meta_loss: Average query loss across all tasks
            meta_acc: Average query accuracy across all tasks
        """
        batch_size = support_images.size(0)
        meta_losses = []
        meta_accs = []
        
        device = support_images.device
        
        self.meta_optimizer.zero_grad()
        
        for i in range(batch_size):
            support_imgs_task = support_images[i]
            support_lbls_task = support_labels[i]
            query_imgs_task = query_images[i]
            query_lbls_task = query_labels[i]
            
            task_model = ConvNet(in_channels=self.model.in_channels, 
                                num_filters=self.model.num_filters, 
                                num_classes=self.model.num_classes)
            task_model.to(device)
            
            for target_param, param in zip(task_model.parameters(), self.model.parameters()):
                target_param.data.copy_(param.data.to(device))
            
            # Inner loop: task-specific adaptation
            for _ in range(self.num_inner_steps):
                support_logits = task_model(support_imgs_task)
                support_loss = F.cross_entropy(support_logits, support_lbls_task)
                
                grads = torch.autograd.grad(support_loss, task_model.parameters(), 
                                           create_graph=True)
                
                for p, g in zip(task_model.parameters(), grads):
                    p.data.sub_(self.inner_lr * g)
            
            # Outer loop: meta-update
            query_logits = task_model(query_imgs_task)
            query_loss = F.cross_entropy(query_logits, query_lbls_task)
            
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_lbls_task).float().mean()
            
            query_loss.backward()
            
            meta_losses.append(query_loss.item())
            meta_accs.append(acc.item())
        
        # Average gradients across all tasks
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / batch_size)
        
        # Update meta-parameters
        self.meta_optimizer.step()
        
        return np.mean(meta_losses), np.mean(meta_accs)
    
    def meta_evaluate(self, support_images, support_labels, query_images, query_labels):
        """
        Performs meta-evaluation on a batch of tasks
        
        Args:
            support_images: Support set images [batch_size, N*K, C, H, W]
            support_labels: Support set labels [batch_size, N*K]
            query_images: Query set images [batch_size, N*Q, C, H, W]
            query_labels: Query set labels [batch_size, N*Q]
            
        Returns:
            avg_loss: Average query loss across all tasks
            avg_acc: Average query accuracy across all tasks
        """
        batch_size = support_images.size(0)
        losses = []
        accs = []
        
        device = support_images.device
        
        for i in range(batch_size):
            support_imgs_task = support_images[i]
            support_lbls_task = support_labels[i]
            query_imgs_task = query_images[i]
            query_lbls_task = query_labels[i]
            
            task_model = ConvNet(in_channels=self.model.in_channels, 
                                num_filters=self.model.num_filters, 
                                num_classes=self.model.num_classes)
            task_model.to(device)

            for target_param, param in zip(task_model.parameters(), self.model.parameters()):
                target_param.data.copy_(param.data.to(device))
                target_param.requires_grad_(True)
            
            # Inner loop: task-specific adaptation (using gradients)
            for _ in range(self.num_inner_steps):
                support_logits = task_model(support_imgs_task)
                support_loss = F.cross_entropy(support_logits, support_lbls_task)
                
                grads = torch.autograd.grad(support_loss, task_model.parameters(), 
                                         create_graph=False, retain_graph=False)
                
                for p, g in zip(task_model.parameters(), grads):
                    p.data.sub_(self.inner_lr * g)
            
            with torch.no_grad():
                query_logits = task_model(query_imgs_task)
                query_loss = F.cross_entropy(query_logits, query_lbls_task)
                
                pred = query_logits.argmax(dim=1)
                acc = (pred == query_lbls_task).float().mean()
                
                losses.append(query_loss.item())
                accs.append(acc.item())
        
        return np.mean(losses), np.mean(accs)