import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import random

from model import Pix2PixModel
from process_data import get_data

# Run the training procedure
def train(model, train_data, validation_data, num_epochs = 10, batch_size = 128, 
         learning_rate=0.0002, device=None, gen=False, visualize=True):
    # Check device
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define optimizers
    gen_optimizer = torch.optim.Adam(model.gen.parameters(), lr=learning_rate)
    discrim_optimizer = torch.optim.Adam(model.discrim.parameters(), lr=learning_rate)
    
    # Training iterations
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    num_batches = len(train_dataloader)
    steps = []
    gen_losses = []
    discrim_losses = []
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            for j in range(2):
                # Get losses, append them
                inputs, labels = data[0].to(device), data[1].to(device)

                if j%2 == 0:
                    loss = model.gen_loss(inputs, labels)
                    print(epoch, i, loss.data.item(), end=' ')
                    steps.append((epoch * num_batches) + i)
                    gen_losses.append(loss.data.item())
                else:
                    loss = model.discrim_loss(inputs, labels)
                    print(loss.data.item())
                    steps.append((epoch * num_batches) + i)
                    discrim_losses.append(loss.data.item())

                # Optimize generator loss and/or discriminator loss
                if j%2 == 0:
                    gen_optimizer.zero_grad()
                    loss.backward()
                    gen_optimizer.step()
                else:
                    discrim_optimizer.zero_grad()
                    loss.backward()
                    discrim_optimizer.step()
        
        if gen:
            generate(model, validation_data, device=device)
    
    if visualize:
        graph_losses(steps, gen_losses, discrim_losses)
    
    return model

# Graph losses
def graph_losses(steps, gen_losses, discrim_losses):
    plt.plot(steps, gen_losses)
    plt.plot(steps, discrim_losses)
    plt.show()

# Run the generation procedure
def generate(model, data, num_examples = 3, device=None):
    # Check device
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Pick random num_examples condition images and generate from those
    figure = plt.figure()
    imgs = random.sample(range(len(data)), num_examples)
    ims, labels, = [], []
    for ind in imgs:
        im, label = data[ind]
        ims.append(im)
        labels.append(label)
    
    inps = torch.from_numpy(np.array(ims).reshape((num_examples, 1, 128, 128))).to(device)
    gen_imgs = model.generate(inps).cpu().detach().numpy()

    for i in range(num_examples):
        figure.add_subplot(num_examples, 3, 3*i + 1)
        plt.title("Text Image")
        plt.axis("off")
        plt.imshow(ims[i][0], cmap="gray")

        figure.add_subplot(num_examples, 3, 3*i + 2)
        plt.title("Handwritten Image")
        plt.axis("off")
        plt.imshow(labels[i][0], cmap="gray")

        figure.add_subplot(num_examples, 3, 3*i + 3)
        plt.title("Generated Image")
        plt.axis("off")
        inp = torch.from_numpy(im.reshape((1, 1, 128, 128))).to(device)
        plt.imshow(gen_imgs[i][0], cmap="gray")

    plt.show()


if __name__ == "__main__":
    # Check the arguments for training or generating
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--gen", action="store_true")
    args = parser.parse_args()

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.train:
        # Create the model, get data
        model = Pix2PixModel(1, 1, device=device)
        if args.load:
            model.load_model("checkpoint")
        train_data, validation_data = get_data()

        # Train
        model = train(model, train_data, validation_data, num_epochs=8, device=device, gen=args.gen)

        # Save model
        model.save_model("checkpoint")
        
    elif args.gen:
        # Create and load the model, get data
        model = Pix2PixModel(1, 1, device=device)
        if args.load:
            model.load_model("checkpoint")
        train_data, validation_data = get_data()

        # Generate
        generate(model, validation_data, device=device)

    else:
        print("Must either include the --train or --gen flag!")
        exit()

