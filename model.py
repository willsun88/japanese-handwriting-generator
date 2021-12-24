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

        # Downsample convolutions as defined in the paper
        self.encoders = [
            DownSampleBlock(in_channels, 64, batchnorm=False), 
            DownSampleBlock(64, 128), 
            DownSampleBlock(128, 256),
            DownSampleBlock(256, 512),
            DownSampleBlock(512, 512),
            DownSampleBlock(512, 512), 
            DownSampleBlock(512, 512),
            DownSampleBlock(512, 512, batchnorm=False)
        ]

        # Upsample convolutions as defined in the paper
        self.decoders = [
            UpSampleBlock(512, 512, dropout=True),
            UpSampleBlock(1024, 512, dropout=True),
            UpSampleBlock(1024, 512, dropout=True),
            UpSampleBlock(1024, 512),
            UpSampleBlock(1024, 256),
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
            DownSampleBlock(input_channels, 64, batchnorm=False),
            DownSampleBlock(64, 128),
            DownSampleBlock(128, 256),
            DownSampleBlock(256, 512)
        ]
        self.out = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        for block in self.model:
            x = self.block(x)
        return self.out(x)