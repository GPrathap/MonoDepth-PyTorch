import os
from PIL import Image

from torch.utils.data import Dataset


class KittiLoader(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        left_dir = os.path.join(root_dir, 'image_02/data/')
        self.left_paths = sorted([os.path.join(left_dir, fname) for fname \
                                  in os.listdir(left_dir)])
        if mode == 'train':
            right_dir = os.path.join(root_dir, 'image_03/data/')
            self.right_paths = sorted([os.path.join(right_dir, fname) for fname \
                                       in os.listdir(right_dir)])
            assert len(self.right_paths) == len(self.left_paths)
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.left_paths)

    def __getitem__(self, idx):
        # size = (256, 512)
        left_image = Image.open(self.left_paths[idx])
        # left_image = left_image.resize(size, Image.ANTIALIAS)
        if self.mode == 'train':
            right_image = Image.open(self.right_paths[idx])
            # right_image = right_image.resize(size, Image.ANTIALIAS)
            sample = {'left_image': left_image, 'right_image': right_image}

            if self.transform:
                sample = self.transform(sample)
                return sample
            else:
                return sample
        else:
            if self.transform:
                left_image = self.transform(left_image)
            return left_image
