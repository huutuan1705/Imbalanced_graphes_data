from torch.utils.data import Dataset

class GrapesDataset(Dataset):
    def __init__(self, args):
        super(GrapesDataset, self).__init__()
        self.dir = args.root_dir
        