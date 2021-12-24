from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob

# Our data
dataset = "ETL8G"

# Get curr path
absPath, _ = os.path.split(os.path.abspath(__file__))

# Get all subfolders of our data
sub_folders = []
for dir, sub_dirs, files in os.walk(absPath+"/data/" + dataset):
    sub_folders.extend(sub_dirs)

# Gather the images and convert them to np arrays
x_data = []
y_data = []
for folder in sub_folders:
    print(folder)
    images = glob.glob(absPath + "/data/" + dataset + "/" + folder + "/*.png")

    # Open the text image
    true_img = Image.open(absPath + "/data/" + dataset + "/" + folder + "/true.png")
    true_arr = np.asarray(true_img)
    true_img.close()
    for f in images:
        # Open and save the image if it isn't the text image
        if "true" not in f:
            curr_img = Image.open(f).convert('RGB')
            y_data.append(np.asarray(curr_img))
            x_data.append(true_arr)
            curr_img.close()

# Convert to np arrays
x_data = np.array(x_data)
y_data = np.array(y_data)
print(x_data.shape, y_data.shape)

with open(absPath + "/data/data.npy", 'wb') as f:
    np.save(f, x_data)
    np.save(f, y_data)
    