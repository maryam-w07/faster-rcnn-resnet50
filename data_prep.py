from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.images = [os.path.join(root, img) for img in os.listdir(root) if img.endswith('.jpg')]

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image

    def __len__(self):
        return len(self.images)

def get_dataloader(root, batch_size=4, transforms=None):
    dataset = CustomDataset(root, transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
