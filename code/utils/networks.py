import torch
import torch.nn as nn
from torch.nn import functional as F

class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""
    def forward(self, x):
        return x * F.sigmoid(x)
    

# We use a deep convolutional generative adversarial network for the images.
class ImageEncoder(nn.Module):
    """
    define the PyTorch module that parametrizes q(z|image).
    This goes from images to the latent z
    
    This is the standard DCGAN architecture.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
    """
    def __init__(self, z_dim):
        super(ImageEncoder, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        #                padding=0, dilation=1, groups=1, bias=True)
        # H_out = floor( (H_in + 2*padding - dilation(kernel_size-1)-1) /
        #                stride+1)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            Swish(),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.Conv2d(128, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            Swish())
        # Here, we define two layers, one to give z_loc and one to give z_scale
        self.z_loc_layer = nn.Sequential(
            # it's 256 * 5 * 5 if input is 64x64.
            nn.Linear(256 * 5 * 5, 512),
            # it's 256 * 9 * 9 if input is 100x100.
            # nn.Linear(256 * 9 * 9, 512),
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, z_dim))
        self.z_scale_layer = nn.Sequential(
            # it's 256 * 5 * 5 if input is 64x64.
            nn.Linear(256 * 5 * 5, 512),
            # it's 256 * 9 * 9 if input is 100x100.
            # nn.Linear(256 * 9 * 9, 512), 
            Swish(),
            nn.Dropout(p=0.1),
            nn.Linear(512, z_dim))
        self.z_dim = z_dim

    def forward(self, image):
        hidden = self.features(image)
        # it's 256 * 5 * 5 if input is 64x64.
        hidden = hidden.view(-1, 256 * 5 * 5)
        # it's 256 * 9 * 9 if input is 100x100.
        #image = image.view(-1, 256 * 9 * 9)
        z_loc = self.z_loc_layer(hidden)
        #add exp so it's always positive
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale
    
class ImageDecoder(nn.Module):
    """
    define the PyTorch module that parametrizes p(image|z).
    This goes from the latent z to the images
    
    This is the standard DCGAN architecture.

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
    """
    def __init__(self, z_dim):
        super(ImageDecoder, self).__init__()
        self.upsample = nn.Sequential(
            # it's 256 * 5 * 5 if input is 64x64.
            nn.Linear(z_dim, 256 * 5 * 5),
            # it's 256 * 9 * 9 if input is 100x100.
            # nn.Linear(z_dim, 256 * 9 * 9),
            Swish())
        self.hallucinate = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            Swish(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            Swish(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            Swish(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False))

    def forward(self, z):
        # the input will be a vector of size |z_dim|
        z = self.upsample(z)
        z = z.view(-1, 256, 5, 5) # it's 256 * 5 * 5 if input is 64x64.
        #z = z.view(-1, 256, 9, 9) # it's 256 * 9 * 9 if input is 100x100.
        # but if 100x100, the output image size is 96x96
        image = self.hallucinate(z) # this is the image
        return image  # NOTE: no sigmoid here. See train.py


# For the other modalities, we use a common network structure
# with two hidden layers for both the encoder and decoder.
# The networks have two outputs: mean and variance.

class Encoder(nn.Module):
    """
    define the PyTorch module that parametrizes q(z|input).
    This goes from inputs to the latent z

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
    """
    def __init__(self, z_dim, input_dim, hidden_dim=512):
        super(Encoder, self).__init__()
        self.net = nn.Linear(input_dim, hidden_dim)
        self.z_loc_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, z_dim))
        self.z_scale_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, z_dim))
        self.z_dim = z_dim

    def forward(self, input):
        hidden = self.net(input)
        z_loc = self.z_loc_layer(hidden)
        z_scale = torch.exp(self.z_scale_layer(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):
    """
    define the PyTorch module that parametrizes p(output|z).
    This goes from the latent z to the output

    @param z_dim: integer
                  size of the tensor representing the latent random variable z
    """
    def __init__(self, z_dim, output_dim, hidden_dim=512):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            Swish())
        self.output_loc_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim))
        self.output_scale_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim))

    def forward(self, z):
        hidden = self.net(z)
        output_loc = self.output_loc_layer(hidden)
        output_scale = torch.exp(self.output_scale_layer(hidden))
        return output_loc, output_scale  # NOTE: no softmax here. See train.py
