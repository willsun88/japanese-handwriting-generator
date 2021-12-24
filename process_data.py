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
        self.x_images = x_images
        self.y_images = y_images
    def __getitem__(self, index):
        return self.x_images[index], self.y_images[index]
    def __len__(self):
        return self.x_images.shape[0]

# Function to get the data, assuming it is stored in the right place
def get_data(percent_train=0.9):
    # Open compressed data
    with open('data/data.npy', 'rb') as f:
        x_data = np.load(f)
        y_data = np.load(f)

    # Shuffle data
    shuffle_indices = np.arange(len(x_data))
    np.random.shuffle(shuffle_indices)
    x_data = x_data[shuffle_indices]
    y_data = y_data[shuffle_indices]

    # Split data into test and train
    split_index = math.floor(percent_train*len(x_data))
    train_x = x_data[:split_index]
    train_y = y_data[:split_index]
    test_x = x_data[split_index:]
    test_y = y_data[split_index:]

    # Return data wrapped in a kanji dataset
    return KanjiDataset(train_x, train_y), KanjiDataset(test_x, test_y)

# A short common sense test if running this file
if __name__ == "__main__":
    train_data, test_data = get_data()

    # Print shapes
    print(train_data.x_images.shape, train_data.y_images.shape)
    print(test_data.x_images.shape, test_data.y_images.shape)

    # Display random image/label pair
    figure = plt.figure()
    im, label = train_data[random.randint(0, len(train_data) - 1)]

    figure.add_subplot(1, 2, 1)
    plt.title("Text Image")
    plt.axis("off")
    plt.imshow(im, cmap="gray")

    figure.add_subplot(1, 2, 2)
    plt.title("Handwritten Image")
    plt.axis("off")
    plt.imshow(label, cmap="gray")

    plt.show()