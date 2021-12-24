import numpy as np
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import argparse

from model import Pix2PixModel
from process_data import get_data

# Run the training procedure
def train(model, train_data, num_epochs = 100, batch_size = 128, learning_rate=0.0002, device=None):
    # Check device
    if device==None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define optimizers
    gen_optimizer = torch.optim.Adam(model.gen.parameters(), lr=learning_rate)
    discrim_optimizer = torch.optim.Adam(model.discrim.parameters(), lr=learning_rate)
    
    # Training iterations
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        for i, data in enumerate(train_dataloader):
            # Call the model
            inputs, labels = data[0].to(device), data[1].to(device)
            gen_out, disc_out = model.call(inputs)

            # Get losses
            gen_loss = model.gen_loss(gen_out, disc_out, labels)
            discrim_loss = model.discrim_loss(gen_out, disc_out, labels)
            print(epoch, i, gen_loss.data.item(), discrim_loss.data.item())

            # Optimize generator loss
            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            # Optimize discriminator loss
            discrim_optimizer.zero_grad()
            discrim_loss.backward()
            discrim_optimizer.step()
    
    return model

# Run the generation procedure
def generate(model):
    pass


if __name__ == "__main__":
    # Check the arguments for training or generating
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--gen", action="store_true")
    args = parser.parse_args()

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the model, get data
    model = Pix2PixModel(1, 1, device=device)
    train_data = get_data()
    
    if args.train:
        model = train(model, train_data, device=device)
        if args.gen:
            pass
    elif args.gen:
        pass
    else:
        print("Must either include the --train or --gen flag!")
        exit()

