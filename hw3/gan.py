import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        in_channels, h, w = in_size

        # Define the convolutional layers
        modules = []
        """channels = [in_channels, 64, 128, 256, 512, 1]
        for i in range(len(channels) - 1):
            modules.append(nn.Conv2d(channels[i], channels[i+1], kernel_size=4, stride=2 if i < 4 else 1, padding=1 if i < 4 else 0))
            if i < len(channels) - 2:  # No activation or batch norm after the last layer
                if i > 0:  # No batch norm after the first layer
                    modules.append(nn.BatchNorm2d(channels[i+1]))
                modules.append(nn.LeakyReLU(0.2, inplace=True))"""
        
        output_channels = 512

        modules = [
            nn.Conv2d(in_channels, 128, 3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(256, 512, 3, padding=2, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(512, output_channels, 3, padding=1, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2),
        ]

        self.cnn = nn.Sequential(*modules)
        self.final_size = self._calc_num_cnn_features(in_size)
        self.fc = nn.Linear(self.final_size, 1)
        # ========================

    def _calc_num_cnn_features(self, in_shape):
        with torch.no_grad():
            x = torch.zeros(1, *in_shape)
            out_shape = self.cnn(x).shape
        return int(np.prod(out_shape))

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        batch_size = x.shape[0]
        features = self.cnn(x)
        features = features.view(batch_size, -1)  # flatten
        y = self.fc(features)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim


        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======
        # Define the transpose convolutional layers using a loop
        """modules = []
        channels = [z_dim, 512, 256, 128, 64, out_channels]
        strides = [1, 2, 2, 2, 2]
        paddings = [0, 1, 1, 1, 1]
        kernel_sizes = [featuremap_size, 4, 4, 4, 4]

        for i in range(len(channels) - 1):
            modules.append(nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            if i < len(channels) - 2:  # No activation or batch norm after the last layer
                modules.append(nn.BatchNorm2d(channels[i+1]))
                modules.append(nn.ReLU(True))
            else:
                modules.append(nn.Tanh())  # Tanh activation for the last layer"""
        
        # Define the initial feature map size
        self.features_c = 512
        self.features_h = featuremap_size
        self.features_w = featuremap_size
        self.features_size = self.features_c * self.features_h * self.features_w
        self.z_to_features = nn.Linear(z_dim, self.features_size)

        # Define the transpose convolutional layers
        modules = [
            nn.ConvTranspose2d(self.features_c, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()  # Output should be in the range [-1, 1]
        ]
        
        self.net = nn.Sequential(*modules)
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        noise = torch.randn(n, self.z_dim, device=device)
        if not with_grad:
            with torch.no_grad():
                samples = self.forward(noise)
        else:
            samples = self.forward(noise)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        batch_size = z.size(0)
        features = self.z_to_features(z).view(batch_size, self.features_c, self.features_h, self.features_w)
        x = self.net(features)
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss. Apply noise to both the real data and the
    #  generated labels.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    device = y_data.device

    # Create noisy labels for real data
    real_labels = torch.full_like(y_data, float(data_label), device=device)
    if label_noise > 0:
        noise = torch.empty_like(real_labels, device=device).uniform_(-label_noise / 2, label_noise / 2)
        real_labels = real_labels + noise

    # Create noisy labels for generated data
    generated_labels = torch.full_like(y_generated, float(1 - data_label), device=device)
    if label_noise > 0:
        noise = torch.empty_like(generated_labels, device=device).uniform_(-label_noise / 2, label_noise / 2)
        generated_labels = generated_labels + noise

    # Compute loss for real data
    loss_data = loss_fn(y_data, real_labels)
    
    # Compute loss for generated data
    loss_generated = loss_fn(y_generated, generated_labels)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()

    # Create labels for the generated data
    target_labels = torch.full_like(y_generated, float(data_label), device=y_generated.device)
    loss = loss_fn(y_generated, target_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: Tensor,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    batch_size = x_data.shape[0]

    fake_data = gen_model.sample(batch_size, with_grad=False)

    # Discriminator predictions
    y_data = dsc_model(x_data)
    y_generated = dsc_model(fake_data.detach())

    # Compute discriminator loss
    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    # Generate fake data for generator update
    fake_data = gen_model.sample(batch_size, with_grad=True)
    y_generated = dsc_model(fake_data)

    # Compute generator loss
    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======

    # Criterion: Save if the generator's avg loss improved
    if len(gen_losses) > 1 and gen_losses[-1] < min(gen_losses[:-1]):
        gen_model = {
            'epoch': len(gen_losses),
            'gen_model_state_dict': gen_model.state_dict(),
            'gen_loss': gen_losses[-1],
            'dsc_losses': dsc_losses,
            'gen_losses': gen_losses
        }
        torch.save(gen_model, checkpoint_file)
        print(f"*** Saved checkpoint {checkpoint_file}")
        saved = True
    # ========================

    return saved
