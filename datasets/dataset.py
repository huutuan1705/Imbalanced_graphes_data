import os
from torch.utils.data import Dataset

class GrapesDataset(Dataset):
    def __init__(self, args, transform=None, mode='train'):
        super(GrapesDataset, self).__init__()
        self.root_dir = args.root_dir
        self.transform = transform
        self.mode = mode
        
        self.train_dir = self.root_dir + "/train"
        self.test_dir = self.root_dir + "/test"
        
        self.train_imgs = sorted(os.listdir(self.train_dir + "/images"))
        self.train_labels = sorted(os.listdir(self.train_dir + "/labels"))
        
        self.test_imgs = sorted(os.listdir(self.test_dir + "/images"))
        self.test_labels = sorted(os.listdir(self.test_dir + "/labels"))
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_imgs)
        return len(self.test_imgs)
    
    def __getitem__(self, idx):
        img_path = os.path.join()
        return 