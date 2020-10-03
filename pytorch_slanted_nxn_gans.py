import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader,TensorDataset
from torchvision import datasets
from torch.autograd import Variable

from numpy import random

import torch.nn as nn
import torch.nn.functional as F
import torch

import pprint as pp

#Parametros

image_folder = "slanted"
os.makedirs(image_folder, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches") # Não descobri pq diminuir o tamanho do batch dá erro
parser.add_argument("--lr", type=float, default=0.0000075, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=120, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=5, help="size of each image dimension") ## Alterado para o tamanho do slanted faces
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

## Usar VGA se disponível
cuda = True if torch.cuda.is_available() else False

### Gerador de dados bons e ruins
def good_face_gen():
    return [[random.uniform(0.8, 1), random.uniform(0, 0.2), random.uniform(0, 0.2)], [random.uniform(0, 0.2), random.uniform(0.8, 1),random.uniform(0, 0.2)],[random.uniform(0, 0.2), random.uniform(0.0, 0.2),random.uniform(0.8, 1)]]

def good_face_gen(face_size):
    return np.apply_along_axis(lambda a: a*random.uniform(0.8, 1.2)/random.uniform(0.8, 1.2),0,np.identity(face_size) ) 

def bad_face_gen():
    return np.array(
        [[random.uniform(0, 0.7), random.uniform(0.3, 1), random.uniform(0, 1)], [random.uniform(0.3, 1), random.uniform(0.2, 0.8), random.uniform(0.02, 0.92)],[random.uniform(0.3, 1),random.uniform(0, 1), random.uniform(0, 1)]])

def bad_face_gen(face_size):
    return np.random.rand(face_size,face_size)


faces_x = np.array([good_face_gen(face_size=opt.img_size)])
faces_y = np.array([1])

for i in range(4000):
    if random.uniform(0, 1) > 0.5:
        faces_x = np.append(faces_x, [good_face_gen(face_size=opt.img_size)], axis=0)
        faces_y = np.append(faces_y, [1], axis=0)
    else:
        faces_x = np.append(faces_x, [bad_face_gen(face_size=opt.img_size)], axis=0)
        faces_y = np.append(faces_y, [0], axis=0)

test_faces_x = np.array([good_face_gen(face_size=opt.img_size)])
test_faces_y = np.array([1])


#####


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 32, normalize=False),
            *block(32, 64),
            *block(64, 128),
            *block(128, 256),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader

### Original
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=opt.batch_size,
#     shuffle=True,
# )

### Slanted - Dataloader de numpy
tensor_x = torch.Tensor(faces_x) # transform to torch tensor
tensor_y = torch.Tensor(faces_y)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your dataset
my_dataloader = DataLoader(my_dataset,batch_size=opt.batch_size,shuffle=True) # create your dataloader


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------


loss_history=[]

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(my_dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(my_dataloader), d_loss.item(), g_loss.item())
        )
            
        ## Armazena o histórico do loss no final de cada época    
        if i==(len(my_dataloader)-1):
            loss_history.append({"epoch": epoch,"d_loss": d_loss.item(),"g_loss": g_loss.item()})

        batches_done = epoch * len(my_dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], image_folder+"/%d.png" % batches_done, nrow=5, normalize=True)
            

for record in loss_history:
    pp.pprint(record,indent=3)