import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_sample(path):
    image = plt.imread(path)
    image_ar = np.array(image)
    image_ar = np.moveaxis(image_ar, -1, 0)
    image_tensor = torch.from_numpy(image_ar)
    return image_tensor


file_list = os.listdir(os.path.join(os.getcwd(), "filtered_data"))

mammal_files = []
for file in file_list:
    if int(file[-6:-4]) in [
        1,
        2,
        3,
        4,
        5,
        41,
        42,
        43,
        44,
        45,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        81,
        82,
        83,
        84,
        85,
    ]:
        mammal_files.append(file)

filtered_data_dir = "filtered_data"


def get_paths(filtered_data_dir):
    anchor_paths = []
    positive_paths = []
    negative_paths = []
    labels = []
    for file in mammal_files:
        label_1 = int(file[-6:-4])

        idx_2 = np.random.randint(0, len(mammal_files))
        second_file = mammal_files[idx_2]
        label_2 = int(second_file[-6:-4])

        if label_1 == label_2:
            anchor_paths.append(os.path.join(os.getcwd(), filtered_data_dir, file))
            positive_paths.append(
                os.path.join(os.getcwd(), filtered_data_dir, second_file)
            )
            random_idx = np.random.randint(0, len(mammal_files))

            third_file = mammal_files[random_idx]
            label_3 = int(third_file[-6:-4])
            while label_3 == label_1:
                random_idx = np.random.randint(0, len(mammal_files))
                third_file = mammal_files[random_idx]
                label_3 = int(third_file[-6:-4])

            negative_paths.append(
                os.path.join(os.getcwd(), filtered_data_dir, third_file)
            )

            labels.append(label_1)
        else:
            anchor_paths.append(os.path.join(os.getcwd(), filtered_data_dir, file))
            negative_paths.append(
                os.path.join(os.getcwd(), filtered_data_dir, second_file)
            )
            random_idx = np.random.randint(0, len(mammal_files))

            third_file = mammal_files[random_idx]
            label_3 = int(third_file[-6:-4])
            while label_3 != label_1:
                random_idx = np.random.randint(0, len(mammal_files))
                third_file = mammal_files[random_idx]
                label_3 = int(third_file[-6:-4])

            positive_paths.append(
                os.path.join(os.getcwd(), filtered_data_dir, third_file)
            )

            labels.append(label_1)
    return anchor_paths, positive_paths, negative_paths, labels, mammal_files


class TrainingDataset(Dataset):
    def __init__(self, anchor_paths, positive_paths, negative_paths):
        self.anchor_paths = anchor_paths
        self.positive_paths = positive_paths
        self.negative_paths = negative_paths

    def __len__(self):
        return len(self.anchor_paths)

    def __getitem__(self, idx):
        anchor = load_sample(self.anchor_paths[idx])
        positive = load_sample(self.positive_paths[idx])
        negative = load_sample(self.negative_paths[idx])
        return anchor, positive, negative


class ValidationDataset(Dataset):
    def __init__(self, pretrained_data):
        self.pretrained_data = (
            pretrained_data  # Initialising validation data with only pretained data
        )

    def __len__(self):
        return len(self.pretrained_data)

    def __getitem__(self, idx):
        sample = load_sample(self.pretrained_data[idx]["image_path"])
        label = self.pretrained_data[idx]["label"]
        return sample, label


class TestingDataset(Dataset):
    def __init__(self, pretrained_data, new_images):
        self.pretrained_data = pretrained_data
        self.new_images = new_images

    def __len__(self):
        return len(self.pretrained_data) + len(self.new_images)

    def __getitem__(self, idx):
        if idx < len(self.pretrained_data):
            sample = load_sample(self.pretrained_data[idx]["image_path"])
            label = self.pretrained_data[idx]["label"]
        else:
            idx -= len(self.pretrained_data)
            sample = load_sample(self.new_images[idx]["image_path"])
            label = self.new_images[idx]["label"]
        return sample, label


anchor_paths, positive_paths, negative_paths, labels, mammal_files = get_paths(
    filtered_data_dir
)

pretrained_data = []

training_dataset = TrainingDataset(anchor_paths, positive_paths, negative_paths)
validation_dataset = ValidationDataset(pretrained_data)
testing_dataset = TestingDataset(pretrained_data)

batch_size = 10
shuffle = True

training_dataloader = DataLoader(
    training_dataset, batch_size=batch_size, shuffle=shuffle
)
validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False)
