from torch.utils.data import Dataset
import os
# Dataset to load image paths and labels.
class ImagePathDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_data=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.max_data = max_data
        for class_idx, class_dir in enumerate(os.listdir(root_dir)):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.image_paths.append(os.path.join(class_path, img_file))
                    self.labels.append(class_idx)
        
        # Limit the data if max_data is set
        if self.max_data is not None:
            self.image_paths = self.image_paths[:self.max_data]
            self.labels = self.labels[:self.max_data]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        return image_path, label
