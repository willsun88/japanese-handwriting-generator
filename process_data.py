from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import random
import math

# Our data
dataset = "ETL8G"

# Get curr path
absPath, _ = os.path.split(os.path.abspath(__file__))

# Dataset class to represent our data, and 
# we will pass an instance to main
class KanjiDataset(Dataset):
    def __init__(self, x_images, y_images):
        self.x_images = x_images.astype(np.float32)
        self.y_images = y_images.astype(np.float32)
    def __getitem__(self, index):
        return self.x_images[index], self.y_images[index]
    def __len__(self):
        return self.x_images.shape[0]

# Function to get the data, assuming it is stored in the right place
def get_data():
    # Open compressed data
    with open('data/data.npy', 'rb') as f:
        x_data = np.load(f)
        y_data = np.load(f)

    # Shuffle data and reshape
    shuffle_indices = np.arange(len(x_data))
    np.random.shuffle(shuffle_indices)
    x_data = x_data[shuffle_indices].reshape(
        (x_data.shape[0], 1, x_data.shape[1], x_data.shape[2]))
    y_data = y_data[shuffle_indices].reshape(
        (y_data.shape[0], 1, y_data.shape[1], y_data.shape[2]))

    # Return data wrapped in a kanji dataset
    return KanjiDataset(x_data, y_data)

# A short common sense test if running this file
if __name__ == "__main__":
    data = get_data()

    # Print shapes
    print(data.x_images.shape, data.y_images.shape)

    # Display random image/label pair
    figure = plt.figure()
    im, label = data[random.randint(0, len(data) - 1)]

    figure.add_subplot(1, 2, 1)
    plt.title("Text Image")
    plt.axis("off")
    plt.imshow(im, cmap="gray")

    figure.add_subplot(1, 2, 2)
    plt.title("Handwritten Image")
    plt.axis("off")
    plt.imshow(label, cmap="gray")

    plt.show()