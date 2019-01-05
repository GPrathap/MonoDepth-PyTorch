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

import numpy as np
#import sklearn.datasets
import torch.nn.functional as F
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

# plot paramsdsad

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (15, 10)


def return_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Monodepth')

    parser.add_argument('--data_dir',
                        help='path to the dataset folder. \
                        It should contain subfolders with following structure:\
                        "image_02/data" for left images and \
                        "image_03/data" for right images'
                        , default="/dataset/model1"
                        )
    parser.add_argument('--val_data_dir',
                        help='path to the validation dataset folder. \
                            It should contain subfolders with following structure:\
                            "image_02/data" for left images and \
                            "image_03/data" for right images',
                        default="/dataset/model1"
                        )
    parser.add_argument('--model_path', help='path to the trained model',
                        default="/root/wakemeup/model")
    parser.add_argument('--output_directory',
                        help='where save dispairities\
                        for tested images',
                        default="/root/wakemeup/output"
                        )
    parser.add_argument('--output_image_directory',
                        help='where save dispairities\
                            for tested images',
                        default="/root/wakemeup/images"
                        )
    parser.add_argument('--input_height', type=int, help='input height',
                        default=256)
    parser.add_argument('--input_width', type=int, help='input width',
                        default=512)
    parser.add_argument('--model', default='resnet18_md',
                        help='encoder architecture: ' +
                             'resnet18_md or resnet50_md ' + '(default: resnet18)'
                             + 'or torchvision version of any resnet model'
                        )
    parser.add_argument('--pretrained', default=False,
                        help='Use weights of pretrained model'
                        )
    parser.add_argument('--mode', default='test',
                        help='mode: train or test (default: train)')
    parser.add_argument('--epochs', default=50,
                        help='number of total epochs to run')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='initial learning rate (default: 1e-4)')
    parser.add_argument('--batch_size', default=2, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--adjust_lr', default=True,
                        help='apply learning rate decay or not\
                        (default: True)'
                        )
    parser.add_argument('--device',
                        default='cuda:0',
                        help='choose cpu or cuda:0 device"'
                        )
    parser.add_argument('--do_augmentation', default=True,
                        help='do augmentation of images or not')
    parser.add_argument('--augment_parameters', default=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
                        help='lowest and highest values for gamma,\
                        brightness and color respectively'
                        )
    parser.add_argument('--print_images', default=False,
                        help='print disparity and image\
                        generated from disparity on every iteration'
                        )
    parser.add_argument('--print_weights', default=False,
                        help='print weights of every layer')
    parser.add_argument('--input_channels', default=3,
                        help='Number of channels in input tensor')
    parser.add_argument('--num_workers', default=4,
                        help='Number of workers in dataloader')
    parser.add_argument('--use_multiple_gpu', default=False)
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--generator_iterations', type=int, default=1)
    parser.add_argument('--discriminator_iterations', type=int, default=1)
    args = parser.parse_args()
    return args


def adjust_learning_rate(optimizer, epoch, learning_rate):
    """Sets the learning rate to the initial LR\
        decayed by 2 every 10 epochs after 30 epoches"""
    if epoch >= 30 and epoch < 40:
        lr = learning_rate / 2
    elif epoch >= 40:
        lr = learning_rate / 4
    else:
        lr = learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def post_process_disparity(disp):
    (_, h, w) = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    (l, _) = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        input_size = self.args.nz
        output_size = input_size*8

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
        self.disp1 = nn.ConvTranspose2d(int(output_size / 64), 2, 4, 2, 1, bias=False)
        self.disp2 = nn.ConvTranspose2d(int(output_size / 32), 2, 4, 2, 1, bias=False)
        self.disp3 = nn.ConvTranspose2d(int(output_size / 16), 2, 4, 2, 1, bias=False)
        self.disp4 = nn.ConvTranspose2d(int(output_size / 8), 2, 4, 2, 1, bias=False)

        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))

        x = F.relu(self.conv4(x))
        disp4 = self.disp4(x)
        x = self.bn4(x)

        x = F.relu(self.conv5(x))
        disp3 = self.disp3(x)
        x = self.bn5(x)

        x = F.relu(self.conv6(x))
        disp2 = self.disp2(x)
        x = self.bn6(x)

        x = F.relu(self.conv7(x))
        disp1 = self.disp1(x)

        return self.tanh(disp1), self.tanh(disp2), self.tanh(disp3), self.tanh(disp4)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disp1 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.disp2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        # self.disp3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.criterion = nn.MSELoss()

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        disp1_predcited = self.disp1(x[3])
        disp2_predcited = self.disp1(x[2])
        disp3_predcited = self.disp1(x[1])
        disp1_error = torch.sqrt(self.criterion(disp1_predcited, x[2]))
        disp2_error = torch.sqrt(self.criterion(disp2_predcited, x[1]))
        disp3_error = torch.sqrt(self.criterion(disp3_predcited, x[0]))
        return disp1_error + disp2_error + disp3_error

class Model:

    def __init__(self, args):
        self.args = args

        # Set up model
        self.device = args.device

        self.fixed_noise = torch.randn(self.args.batch_size, self.args.nz, 1, 1, device=self.device)
        self.real_label = 1
        self.fake_label = 0

        self.ngpu = int(self.args.ngpu)
        self.nz = int(self.args.nz)
        self.ngf = int(self.args.ngf)
        self.ndf = int(self.args.ndf)

        self.model_generator = Generator(self.args).to(self.device)
        self.model_discriminator = Discriminator().to(self.device)

        if self.args.netG != '':
            self.model_generator.load_state_dict(torch.load(self.args.netG))
        # if self.args.netG != '':
        #     self.model_generator.load_state_dict(torch.load(self.args.netG))
        # print(self.model_generator)

        self.model_estimator = get_model(args.model, input_channels=args.input_channels,
                                             pretrained=args.pretrained)
        self.model_estimator = self.model_estimator.to(self.device)

        # self.one = torch.FloatTensor([1])
        # self.mone = self.one * -1
        # self.one = self.one.to(self.device)
        # self.mone = self.mone.to(self.device)

        if args.mode == 'train':
            self.loss_function = MonodepthLoss(
                n=4,
                SSIM_w=0.85,
                disp_gradient_w=0.1, lr_w=1).to(self.device)
            self.optimizer_estimator = optim.Adam(self.model_estimator.parameters(), lr=args.learning_rate)
            self.optimizer_generator = optim.Adam(self.model_generator.parameters(), lr=args.learning_rate)
            self.optimizer_discriminator = optim.Adam(self.model_discriminator.parameters(), lr=args.learning_rate)

            self.val_n_img, self.val_loader = prepare_dataloader(args.val_data_dir, args.mode,
                                                                 args.augment_parameters,
                                                                 False, args.batch_size,
                                                                 (args.input_height, args.input_width),
                                                                 args.num_workers)
        else:
            self.model_estimator.load_state_dict(torch.load(args.model_path))
            args.augment_parameters = None
            args.do_augmentation = False
            args.batch_size = 1

        # Load data
        self.output_directory = args.output_directory
        self.input_height = args.input_height
        self.input_width = args.input_width

        self.n_img, self.loader = prepare_dataloader(args.data_dir, args.mode, args.augment_parameters,
                                                     args.do_augmentation, args.batch_size,
                                                     (args.input_height, args.input_width),
                                                     args.num_workers)

        if 'cuda' in self.device:
            torch.cuda.synchronize()

    def train(self):
        losses_real = []
        losses_fake = []
        val_losses = []
        best_loss_real = float('Inf')
        best_val_loss = float('Inf')

        running_val_loss = 0.0
        running_loss_generator = 0.0

        self.model_estimator.eval()
        for data in self.val_loader:
            data = to_device(data, self.device)
            left = data['left_image']
            right = data['right_image']
            disps_real = self.model_estimator(left)
            loss, _ = self.loss_function(disps_real['disparity'], [left, right])
            # for i in range(len(loss)):
            #     val_losses.append(loss[i].item())
            #     running_val_loss += loss[i].item()
            val_losses.append(loss.item())
            running_val_loss += loss.item()
            # running_val_loss += loss.item()

        running_val_loss = running_val_loss / (self.val_n_img / self.args.batch_size)
        print('Val_loss:', running_val_loss)

        for epoch in range(self.args.epochs):
            if self.args.adjust_lr:
                adjust_learning_rate(self.optimizer_discriminator, epoch, self.args.learning_rate)
            c_time = time.time()
            running_loss = 0.0
            running_loss_generator = 0.0
            self.model_estimator.train()
            self.model_generator.train()
            self.model_discriminator.train()
            iterator = 0
            for data in self.loader:
                # Load data
                iterator = iterator + 1
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                if(left.shape[0]!=self.args.batch_size):
                    continue

                # (1) Update D network
                ###########################
                for p in self.model_estimator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update
                for p in self.model_discriminator.parameters():  # reset requires_grad
                    p.requires_grad = True  # they are set to False below in netG update

                for i in xrange(self.args.discriminator_iterations):

                    self.model_estimator.zero_grad()
                    self.model_discriminator.zero_grad()
                    self.optimizer_generator.zero_grad()

                    disps_real = self.model_estimator(left)
                    loss_real_1, _ = self.loss_function(disps_real['disparity'], [left, right])
                    loss_real_2 = self.model_discriminator(disps_real['disparity'])

                    loss_real = loss_real_1 + loss_real_2
                    # loss_real.backward(self.mone)
                    disc_real = loss_real.mean()

                    noise = torch.randn(self.args.batch_size, self.nz, 1, 1, device=self.device)
                    disps_fake = self.model_generator(noise)
                    # disps_fake = self.model_estimator(left)
                    loss_fake_1, _ = self.loss_function(disps_fake, [left, right])
                    loss_fake_2 = self.model_discriminator(disps_fake)

                    loss_fake = loss_fake_1 + loss_fake_2
                    # loss_fake.backward()
                    disc_fake = loss_fake.mean()

                    d_cost = disc_fake - disc_real
                    d_cost.backward()

                    running_loss += d_cost.item()
                    self.optimizer_discriminator.step()
                    self.optimizer_estimator.step()
                ############################
                # (2) Update G network
                ###########################
                for p in self.model_estimator.parameters():
                    p.requires_grad = False  # to avoid computation
                for p in self.model_discriminator.parameters():
                    p.requires_grad = False  # to avoid computation

                self.model_generator.zero_grad()

                noise = torch.randn(self.args.batch_size, self.nz, 1, 1, device=self.device)
                disps_fake = self.model_generator(noise)
                loss_fake = self.model_discriminator(disps_fake)
                loss_fake.backward()
                g_cost = -loss_fake
                running_loss_generator += g_cost
                self.optimizer_generator.step()

                # # One optimization iteration
                # self.optimizer_discriminator.zero_grad()
                # disps_real = self.model_estimator(left)
                # loss, _ = self.loss_function(disps_real['disparity'], [left, right])
                # loss.backward()
                # disc_real = loss.mean()
                #
                # self.optimizer_discriminator.step()
                # losses_real.append(loss.item())
                # # ---------------------TRAIN G------------------------
                # for p in self.model_estimator.parameters():
                #     p.requires_grad_(False)  # freeze D
                #
                # gen_cost = None
                # feature_distance = 0
                # # print('---train G elapsed time: %d'.format(end - start))
                #
                # # ---------------------TRAIN D------------------------
                # for p in self.model_generator.parameters():  # reset requires_grad
                #     p.requires_grad_(True)  # they are set to False below in training G
                # for i in range(self.args.discriminator_iterations):
                #     # print("Critic iter: " + str(i))
                #
                #     start = timer()
                #     self.model_estimator.zero_grad()
                #
                #     # gen fake data and load real data
                #     noise = torch.randn(self.args.batch_size, self.nz, 1, 1, device=self.device)
                #     with torch.no_grad():
                #         noisev = noise  # totally freeze G, training D
                #     fake_data = self.model_generator(noisev).detach()
                #
                #     # train ith real data
                #     disps_real = self.model_estimator(left)
                #     loss_real, _ = self.loss_function(disps_real['disparity'], [left, right])
                #     disc_real = loss_real.mean()
                #     self.label = torch.full((self.args.batch_size,), self.real_label, device=self.device)
                #     disc_real_image = self.criterion(disps_real['classification'], self.label)
                #     disc_real_image_loss = disc_real_image.mean()
                #
                #     # train with fake data
                #     disps_fake = self.model_estimator(fake_data)
                #     # loss_fake, _ = self.loss_function(disps_fake['disparity'], [fake_data, right])
                #     # disc_fake = loss_fake.mean()
                #     self.label.fill_(self.fake_label)
                #     disc_fake_image = self.criterion(disps_fake['classification'], self.label)
                #     disc_fake_image_loss = disc_fake_image.mean()
                #
                #     feature_distance = torch.norm(disps_real['feature_map'] - disps_fake['feature_map'], 2)
                #
                #     # self.showMemoryUsage(0)
                #
                #     # final disc cost
                #     disc_cost = disc_real + disc_real_image_loss + disc_fake_image_loss
                #     disc_cost = Variable(disc_cost, requires_grad=True)
                #
                #     disc_cost.backward()
                #     self.optimizer_discriminator.step()
                #
                #     running_loss += disc_cost.item()
                #
                # for i in range(self.args.generator_iterations):
                #     # print("Generator iters: " + str(i))
                #     self.optimizer_generator.zero_grad()
                #     noise = torch.randn(self.args.batch_size, self.nz, 1, 1, device=self.device)
                #     noise.requires_grad_(True)
                #     fake_data = self.model_generator(noise)
                #     disps_fake = self.model_estimator(fake_data)
                #     # loss_fake, _ = self.loss_function(disps_fake['disparity'], [fake_data, right])
                #     self.label.fill_(self.real_label)
                #     disc_real_image = self.criterion(disps_fake['classification'], self.label)
                #     disc_real_image_loss = disc_real_image.mean()
                #     # disc_fake = loss_fake.mean()
                #     # loss_fake.backward(self.mone)
                #     generator_total_loss = disc_real_image_loss + feature_distance
                #     generator_cost = Variable(generator_total_loss, requires_grad=True)
                #     generator_cost.backward()
                #
                #     running_loss_generator += generator_cost.item()
                #     # loss_fake.backward()

                # if epoch % 1 == 0:
                #     print("Iteration " + str(iterator) + " : loss " + str(disc_real))
                    # fake = self.model_generator(self.fixed_noise)
                    # vutils.save_image(fake[0].detach(),
                    #                   '%s/fake_samples_epoch_%03d.png' % (self.args.output_image_directory, epoch),
                    #                   normalize=True)

                # Print statistics
                if self.args.print_weights:
                    j = 1
                    for (name, parameter) in self.model_estimator.named_parameters():
                        if name.split(sep='.')[-1] == 'weight':
                            plt.subplot(5, 9, j)
                            plt.hist(parameter.data.view(-1))
                            plt.xlim([-1, 1])
                            plt.title(name.split(sep='.')[0])
                            j += 1
                    plt.show()

                if self.args.print_images:
                    print('disp_left_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_left_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('left_est[0]')
                    plt.imshow(np.transpose(self.loss_function \
                                            .left_est[0][0, :, :, :].cpu().detach().numpy(),
                                            (1, 2, 0)))
                    plt.show()
                    print('disp_right_est[0]')
                    plt.imshow(np.squeeze(
                        np.transpose(self.loss_function.disp_right_est[0][0,
                                     :, :, :].cpu().detach().numpy(),
                                     (1, 2, 0))))
                    plt.show()
                    print('right_est[0]')
                    plt.imshow(np.transpose(self.loss_function.right_est[0][0,
                                            :, :, :].cpu().detach().numpy(), (1, 2,
                                                                              0)))
                    plt.show()

                # for i in range(len(loss_real)):
                #     running_loss += loss_real[i].item()

            running_val_loss = 0.0
            self.model_estimator.eval()
            for data in self.val_loader:
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                disps = self.model_estimator(left)
                loss, _ = self.loss_function(disps['disparity'], [left, right])
                # for i in range(len(loss)):
                #     val_losses.append(loss[i].item())
                #     running_val_loss += loss[i].item()
                running_val_loss += loss.item()

            # Estimate loss per image
            running_loss /= self.n_img / self.args.batch_size
            running_val_loss /= self.val_n_img / self.args.batch_size
            running_loss_generator /= self.val_n_img / self.args.batch_size
            print(
                'Epoch:',
                epoch + 1,
                'discriminator_loss:',
                running_loss,
                'generator_loss',
                running_loss_generator.item(),
                'val_loss:',
                running_val_loss,
                'time:',
                round(time.time() - c_time, 3),
                's',
            )
            self.save(self.args.model_path[:-4] + '_last.pth')
            if running_val_loss < best_val_loss:
                self.save(self.args.model_path[:-4] + '_cpt.pth')
                best_val_loss = running_val_loss
                print('Model_saved')

        print('Finished Training. Best loss:', best_loss_real)
        self.save(self.args.model_path)


    def save(self, path):
        torch.save(self.model_estimator.state_dict(), path)

    def load(self, path):
        self.model_estimator.load_state_dict(torch.load(path))

    def showMemoryUsage(self, device):
        gpu_stats = gpustat.GPUStatCollection.new_query()
        item = gpu_stats.jsonify()["gpus"][device]
        print('Used/total: ' + "{}/{}".format(item["memory.used"], item["memory.total"]))

    def test(self):
        self.model_estimator.eval()
        disparities = np.zeros((self.n_img,
                                self.input_height, self.input_width),
                               dtype=np.float32)
        disparities_pp = np.zeros((self.n_img,
                                   self.input_height, self.input_width),
                                  dtype=np.float32)
        with torch.no_grad():
            for (i, data) in enumerate(self.loader):
                # Get the inputs
                data = to_device(data, self.device)
                left = data.squeeze()
                # Do a forward pass
                logicstics = self.model_estimator(left)
                disps =  logicstics['disparity']
                disp = disps[0][:, 0, :, :].unsqueeze(1)
                disparities[i] = disp[0].squeeze().cpu().numpy()
                disparities_pp[i] = \
                    post_process_disparity(disps[0][:, 0, :, :] \
                                           .cpu().numpy())

        np.save(self.output_directory + '/disparities.npy', disparities)
        np.save(self.output_directory + '/disparities_pp.npy',
                disparities_pp)
        print('Finished Testing')


def main():
    args = return_arguments()
    if args.mode == 'train':
        model = Model(args)
        model.train()
    elif args.mode == 'test':
        model_test = Model(args)
        model_test.test()

if __name__ == '__main__':
    main()
