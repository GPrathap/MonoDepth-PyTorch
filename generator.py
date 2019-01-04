from __future__ import division
import argparse
import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.utils as vutils
# custom modules
import time
import functools
import argparse
import torch.nn.functional as F

import numpy as np
#import sklearn.datasets

from tensorboardX import SummaryWriter
from torch.autograd import Variable

import pdb
import gpustat


import torch
import torchvision
from torch import nn
from torch import autograd
from torch import optim
from torchvision import transforms, datasets
from torch.autograd import grad
from timeit import default_timer as timer

import torch.nn.init as init


from loss import MonodepthLoss
from utils import get_model, to_device, prepare_dataloader


class Generator_zero(nn.Module):

    def __init__(self):
        super(Generator_zero, self).__init__()

        input_size = 100
        output_size = 100*8

        self.conv1 = nn.ConvTranspose2d(input_size, output_size, (2, 4), 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)

        # state size. (ngf*8) x 2 x 4
        self.conv2 = nn.ConvTranspose2d(output_size, int(output_size / 2), 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_size / 2))

        # state size. (ngf*4) x 4 x 8
        self.conv3 = nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_size / 4))

        # state size. (ngf*2) x 8 x 16
        self.conv4 = nn.ConvTranspose2d(int(output_size / 4), int(output_size / 8), 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(output_size / 8))

        # state size. (ngf*2) x 16 x 32
        self.conv5 = nn.ConvTranspose2d(int(output_size / 8), int(output_size / 16), 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(output_size / 16))

        # state size. (ngf*2) x 32 x 64
        self.conv6 = nn.ConvTranspose2d(int(output_size / 16), int(output_size / 32), 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(int(output_size / 32))

        # state size. (ngf*2) x 64 x 128
        self.conv7 = nn.ConvTranspose2d(int(output_size / 32), int(output_size / 64), 4, 2, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(int(output_size / 64))

        # state size. (ngf*2) x 128 x 256
        self.conv8 = nn.ConvTranspose2d(int(output_size / 64), 2, 4, 2, 1, bias=False)
        self.output = nn.Tanh()

    def forward(self, x, condi_x=None):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.conv8(x)
        return self.output(x)



class Generator_one(nn.Module):

    def __init__(self):
        super(Generator_one, self).__init__()

        input_size = 100
        output_size = 100*6

        self.conv1 = nn.ConvTranspose2d(input_size, output_size, (2, 4), 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)

        # state size. (ngf*8) x 2 x 4
        self.conv2 = nn.ConvTranspose2d(output_size, int(output_size / 2), 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_size / 2))

        # state size. (ngf*4) x 4 x 8
        self.conv3 = nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_size / 4))

        # state size. (ngf*2) x 8 x 16
        self.conv4 = nn.ConvTranspose2d(int(output_size / 4), int(output_size / 8), 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(output_size / 8))

        # state size. (ngf*2) x 16 x 32
        self.conv5 = nn.ConvTranspose2d(int(output_size / 8), int(output_size / 16), 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(output_size / 16))

        # state size. (ngf*2) x 32 x 64
        self.conv6 = nn.ConvTranspose2d(int(output_size / 16), int(output_size / 32), 4, 2, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(int(output_size / 32))

        # state size. (ngf*2) x 128 x 256
        self.conv8 = nn.ConvTranspose2d(int(output_size / 32), 2, 4, 2, 1, bias=False)
        self.output = nn.Tanh()

    def forward(self, x, condi_x=None):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.conv8(x)
        return self.output(x)

class Generator_two(nn.Module):

    def __init__(self):
        super(Generator_two, self).__init__()

        input_size = 100
        output_size = 100*4

        self.conv1 = nn.ConvTranspose2d(input_size, output_size, (2, 4), 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)

        # state size. (ngf*8) x 2 x 4
        self.conv2 = nn.ConvTranspose2d(output_size, int(output_size / 2), 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_size / 2))

        # state size. (ngf*4) x 4 x 8
        self.conv3 = nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_size / 4))

        # state size. (ngf*2) x 8 x 16
        self.conv4 = nn.ConvTranspose2d(int(output_size / 4), int(output_size / 8), 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(output_size / 8))

        # state size. (ngf*2) x 16 x 32
        self.conv5 = nn.ConvTranspose2d(int(output_size / 8), int(output_size / 16), 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(int(output_size / 16))

        # state size. (ngf*2) x 128 x 256
        self.conv8 = nn.ConvTranspose2d(int(output_size / 16), 2, 4, 2, 1, bias=False)
        self.output = nn.Tanh()

    def forward(self, x, condi_x=None):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.conv8(x)
        return self.output(x)

class Generator_three(nn.Module):

    def __init__(self):
        super(Generator_three, self).__init__()

        input_size = 100
        output_size = 100*2

        self.conv1 = nn.ConvTranspose2d(input_size, output_size, (2, 4), 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(output_size)

        # state size. (ngf*8) x 2 x 4
        self.conv2 = nn.ConvTranspose2d(output_size, int(output_size / 2), 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(int(output_size / 2))

        # state size. (ngf*4) x 4 x 8
        self.conv3 = nn.ConvTranspose2d(int(output_size / 2), int(output_size / 4), 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(int(output_size / 4))

        # state size. (ngf*2) x 8 x 16
        self.conv4 = nn.ConvTranspose2d(int(output_size / 4), int(output_size / 8), 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(int(output_size / 8))

        # state size. (ngf*2) x 128 x 256
        self.conv8 = nn.ConvTranspose2d(int(output_size / 8), 2, 4, 2, 1, bias=False)
        self.output = nn.Tanh()

    def forward(self, x, condi_x=None):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.conv8(x)
        return self.output(x)


class Discriminator_zero(nn.Module):
    def __init__(self):
        super(Discriminator_zero, self).__init__()
        self.fc1 = nn.Linear(2*256*512, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 2*256*512)
        #x = self.drop1(F.leaky_relu(self.fc1(x)))
        #x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # wgan format
        # x = self.fc3(x)
        # x = x.mean(0).view(1)

        return x

class Discriminator_one(nn.Module):
    def __init__(self):
        super(Discriminator_one, self).__init__()
        self.fc1 = nn.Linear(2*256*128, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 2*256*128)
        #x = self.drop1(F.leaky_relu(self.fc1(x)))
        #x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # wgan format
        # x = self.fc3(x)
        # x = x.mean(0).view(1)

        return x

class Discriminator_two(nn.Module):
    def __init__(self):
        super(Discriminator_two, self).__init__()
        self.fc1 = nn.Linear(2*128*64, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 2*128*64)
        #x = self.drop1(F.leaky_relu(self.fc1(x)))
        #x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # wgan format
        # x = self.fc3(x)
        # x = x.mean(0).view(1)

        return x

class Discriminator_three(nn.Module):
    def __init__(self):
        super(Discriminator_three, self).__init__()
        self.fc1 = nn.Linear(2*64*32, 600)
        self.fc2 = nn.Linear(600, 600)
        self.fc3 = nn.Linear(600, 1)
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, x, condi_x=None):
        x = x.view(-1, 2*64*32)
        #x = self.drop1(F.leaky_relu(self.fc1(x)))
        #x = self.drop2(F.leaky_relu(self.fc2(x)))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # wgan format
        # x = self.fc3(x)
        # x = x.mean(0).view(1)

        return x


class Generator():
    def __init__(self, use_gpu=True, n_channel=3):
        self.n_level = 4
        self.n_channel = n_channel
        self.use_gpu = use_gpu
        self.Gen_models = []
        self.Dis_models = []
        # G zero and G one both inputs contain condition information
        G_model0 = Generator_zero()
        if use_gpu: G_model0 = G_model0.cuda()
        self.Gen_models.append(G_model0)

        G_model1 = Generator_one()
        if use_gpu: G_model1 = G_model1.cuda()
        self.Gen_models.append(G_model1)

        # G two inputs have no condition information
        G_model2 = Generator_two()
        if use_gpu: G_model2 = G_model2.cuda()
        self.Gen_models.append(G_model2)

        # G two inputs have no condition information
        G_model3 = Generator_two()
        if use_gpu: G_model3 = G_model3.cuda()
        self.Gen_models.append(G_model3)

        # D zero and D one both inputs contain condition information
        D_model0 = Discriminator_zero()
        if use_gpu: D_model0 = D_model0.cuda()
        self.Dis_models.append(D_model0)

        D_model1 = Discriminator_one()
        if use_gpu: D_model1 = D_model1.cuda()
        self.Dis_models.append(D_model1)

        # D two inputs have no condition information
        D_model2 = Discriminator_two()
        if use_gpu: D_model2 = D_model2.cuda()
        self.Dis_models.append(D_model2)

        # D two inputs have no condition information
        D_model3 = Discriminator_three()
        if use_gpu: D_model3 = D_model3.cuda()
        self.Dis_models.append(D_model3)

        # print(self.Gen_models)

#     def generate(self, batchsize, get_level=None, generator=False):
#         """Generate images from LAPGAN generators"""
#         for G in self.Gen_models:
#             G.eval()
#
#         self.outputs = []
#         self.generator_outputs = []
#         for level in range(self.n_level):
#             Gen_model = self.Gen_models[self.n_level - level - 1]
#
#             # generate noise
#             if level == 0:
#                 self.noise_dim = 100
#             elif level == 1:
#                 self.noise_dim = 16 * 16
#             else:
#                 self.noise_dim = 32 * 32
#             noise = Variable(gen_noise(batchsize, self.noise_dim))
#             if self.use_gpu:
#                 noise = noise.cuda()
#
#             x = []
#             if level == 0:
#                 # directly generate images
#                 output_imgs = Gen_model.forward(noise)
#                 if self.use_gpu:
#                     output_imgs = output_imgs.cpu()
#                 output_imgs = output_imgs.data.numpy()
#                 x.append(output_imgs)
#                 self.generator_outputs.append(output_imgs)
#             else:
#                 # upsize
#                 input_imgs = np.array([[cv2.pyrUp(output_imgs[i, j, :])
#                                         for j in range(self.n_channel)]
#                                        for i in range(batchsize)])
#                 condi_imgs = Variable(torch.Tensor(input_imgs))
#                 if self.use_gpu:
#                     condi_imgs = condi_imgs.cuda()
#
#                 # generate images with extra information
#                 residual_imgs = Gen_model.forward(noise, condi_imgs)
#                 if self.use_gpu:
#                     residual_imgs = residual_imgs.cpu()
#                 output_imgs = residual_imgs.data.numpy() + input_imgs
#                 self.generator_outputs.append(residual_imgs.data.numpy())
#                 x.append(output_imgs)
#
#             self.outputs.append(x[-1])
#
#         if get_level is None:
#             get_level = -1
#
#         x = self.outputs[0]
#         t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
#         t[:, :, :x.shape[2], :x.shape[3]] = x
#         result_imgs = t
#         x = self.outputs[1]
#         t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
#         t[:, :, :x.shape[2], :x.shape[3]] = x
#         result_imgs = np.concatenate([result_imgs, t], axis=0)
#         x = self.outputs[2]
#         t = np.zeros(batchsize * self.n_channel * 32 * 32).reshape(batchsize, self.n_channel, 32, 32)
#         t[:, :, :x.shape[2], :x.shape[3]] = x
#         result_imgs = np.concatenate([result_imgs, t], axis=0)
#         # result_imgs = torch.from_numpy(result_imgs)
#         # torch.clamp(result_imgs, min=-1, max=1)
#         # result_imgs = result_imgs.numpy()
#         # result_imgs = (result_imgs+1)/2
#         # result_imgs = 1 - result_imgs
#         return result_imgs
#
#
# def gen_noise(n_instance, n_dim):
#     # return torch.Tensor(np.random.uniform(low=-1.0, high=1.0, size=(n_instance, n_dim)))
#     return torch.Tensor(np.random.normal(loc=0, scale=0.02, size=(n_instance, n_dim)))
#
#
#
#
#
#
#
#
