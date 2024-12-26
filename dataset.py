import torch
import numpy as np

class Dataset1(torch.utils.data.Dataset):
    def __init__(self, images, tags, labels, test=None):
        self.test = test
        training_index = np.arange(tags.shape[0])
        self.train_images = torch.from_numpy(np.array(images))
        self.train_tags = torch.from_numpy(np.array(tags))
        self.train_labels = torch.from_numpy(np.array(labels))

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.train_images[idx], self.train_tags[idx], self.train_labels[idx]

