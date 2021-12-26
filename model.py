import torch
import torch.nn as nn

class UpSampleBlock(nn.Module):
    """
    A class to handle upsampling. Uses convolutions, batchnorm (default true), 
    relu (default true), and dropout (default false).
    """
    def __init__(self, in_channels, out_channels, ker=4, stride=2, pad=1, relu=True, 
                norm=True, dropout=False):
        super().__init__()
        self.relu = relu
        self.norm = norm
        self.dropout = dropout

        # Define model
        self.convLayer = nn.ConvTranspose2d(in_channels, out_channels, ker, 
                                            stride, pad)
        if self.norm:
            self.batchNorm = nn.BatchNorm2d(out_channels)
        if self.relu:
            self.activ = nn.ReLU(True)
        if self.dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.convLayer(x)
        if self.norm:
            x = self.batchNorm(x)
        if self.dropout:
            x = self.drop(x)
        if self.relu:
            x = self.activ(x)
        return x

class DownSampleBlock(nn.Module):
    """
    A class to handle downsampling. Uses convolutions, batchnorm (default true), 
    and leaky relu (default true).
    """
    def __init__(self, in_channels, out_channels, ker=4, stride=2, pad=1, relu=True, norm=True):
        super().__init__()
        self.relu = relu
        self.norm = norm

        # Define model
        self.convLayer = nn.Conv2d(in_channels, out_channels, ker, stride, pad)
        if self.norm:
            self.batchNorm = nn.BatchNorm2d(out_channels)
        if self.relu:
            self.activ = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.convLayer(x)
        if self.norm:
            x = self.batchNorm(x)
        if self.relu:
            x = self.activ(x)
        return x

class Generator(nn.Module):
    """
    Implemetation of the generator. This is a U-Net model, so it includes skip connections.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder structure as defined in the paper, with a slight
        # modification to match the size of our input
        self.encoders = [
            DownSampleBlock(in_channels, 64, norm=False), 
            DownSampleBlock(64, 128), 
            DownSampleBlock(128, 256),
            DownSampleBlock(256, 256),
            DownSampleBlock(256, 256),
            DownSampleBlock(256, 256), 
            DownSampleBlock(256, 256, norm=False)
        ]

        # Decoder structure as defined in the paper, with a slight
        # modification to match the size of our input
        self.decoders = [
            UpSampleBlock(256, 256, dropout=True),
            UpSampleBlock(512, 256, dropout=True),
            UpSampleBlock(512, 256),
            UpSampleBlock(512, 256),
            UpSampleBlock(512, 128),
            UpSampleBlock(256, 64)
        ]
        self.last_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        # Apply skip connections
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        skips = skips[:-1][::-1]
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips):
            x = decoder(x)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        x = self.last_conv(x)
        return nn.Tanh()(x)

class Discriminator(nn.Module):
    """
    Implemetation of the discriminator. This is a PatchGAN model.
    """
    def __init__(self, input_channels):
        super().__init__()
        self.model = [
            DownSampleBlock(input_channels, 64, norm=False),
            DownSampleBlock(64, 128)
        ]
        self.model = nn.ModuleList(self.model)
        self.out = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        for block in self.model:
            x = block(x)
        return self.out(x)

class Pix2PixModel(object):
    """
    Class that represents the whole model. Includes loss and generation.
    """
    def __init__(self, in_channels, out_channels, lambda_recon=200, device=None):
        if device==None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.lambda_recon = lambda_recon
        self.gen = Generator(in_channels, out_channels)
        self.discrim = Discriminator(in_channels + out_channels)

        # Initialize weights
        self.gen = self.gen.apply(Pix2PixModel.weights_init)
        self.discrim = self.discrim.apply(Pix2PixModel.weights_init)

        # Move models to device
        self.gen.to(self.device)
        self.discrim.to(self.device)

        # Initialize loss calculations
        self.adversarial_criterion = nn.BCEWithLogitsLoss()
        self.recon_criterion = nn.L1Loss()
    
    
    def weights_init(m):
        # Initialize weights as defined in the paper
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    
    def generate_cond(self, cond_inp):
        gen_out = self.gen(cond_inp)
        return gen_out

    def gen_loss(self, cond_inp, true_images):
        gen_out = self.gen(cond_inp)
        disc_out = self.discrim(gen_out, cond_inp)

        adversarial_loss = self.adversarial_criterion(disc_out, torch.ones_like(disc_out))
        recon_loss = self.recon_criterion(gen_out, true_images)

        return adversarial_loss + self.lambda_recon*recon_loss

    def discrim_loss(self, cond_inp, true_images):
        gen_out = self.gen(cond_inp).detach()
        disc_out = self.discrim(gen_out, cond_inp)
        true_out = self.discrim(true_images, cond_inp)

        gen_loss = self.adversarial_criterion(disc_out, torch.zeros_like(disc_out))
        true_loss = self.adversarial_criterion(true_out, torch.ones_like(true_out))

        return (gen_loss + true_loss)/2

    def save_model(self, filename):
        torch.save({
            'gen_dict': self.gen.state_dict(), 
            'discrim_dict': self.discrim.state_dict()
        }, filename)

    def load_model(self, filename):
        weights = torch.load(filename)
        self.gen.load_state_dict(weights['gen_dict'])
        self.discrim.load_state_dict(weights['discrim_dict'])