import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from model import *
from loss import *
from dataloader import *

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    opt = parser.parse_args()
    return opt

def train(opt):
    # Init Model
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    discriminator.train()

    # Load Dataset
    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader('train', opt, train_dataset)

    # test_dataset = MNISTDataset('test')
    # test_data_loader = MNISTDataloader('test', opt, test_dataset)

    # Set Optimizer
    optim_gen = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optim_dis = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    # Set Loss
    loss = Loss()

    writer = SummaryWriter()

    for epoch in range(opt.epoch):
        for i in range(len(train_data_loader.data_loader)):
            step = epoch * len(train_data_loader.data_loader) + i + 1
            # load dataset only batch_size
            image, label = train_data_loader.next_batch()
            image = image.cuda()

            # train discriminator
            optim_dis.zero_grad()

            noise = Variable(torch.randn(opt.batch_size, 100)).cuda()
            gen = generator(noise)

            validity_real = discriminator(image)
            loss_dis_real = loss(validity_real, Variable(torch.ones(opt.batch_size,1)).cuda())

            validity_fake = discriminator(gen.detach())
            loss_dis_fake = loss(validity_fake, Variable(torch.zeros(opt.batch_size,1)).cuda())

            loss_dis = (loss_dis_real + loss_dis_fake) / 2
            loss_dis.backward()
            optim_dis.step()

            # train generator
            generator.train()
            optim_gen.zero_grad()

            noise = Variable(torch.randn(opt.batch_size, 100)).cuda()
            
            gen = generator(noise)
            validity = discriminator(gen)
            
            loss_gen = loss(validity, Variable(torch.ones(opt.batch_size,1)).cuda())
            loss_gen.backward()
            optim_gen.step()

            writer.add_scalar('loss/gen', loss_gen, step)
            writer.add_scalar('loss/dis', loss_dis, step)
            writer.add_scalar('loss/dis_real', loss_dis_real, step)
            writer.add_scalar('loss/dis_fake', loss_dis_fake, step)
            
            if step % opt.display_step == 0:
                writer.add_images('image', image[0][0], step, dataformats="HW")
                writer.add_images('result', gen[0][0], step, dataformats="HW")

                print('[Epoch {}] G_loss : {:.2} | D_loss : {:.2}'.format(epoch + 1, loss_gen, loss_dis))
                
                generator.eval()
                z = Variable(torch.randn(9, 100)).cuda()
                sample_images = generator(z)
                grid = make_grid(sample_images, nrow=3, normalize=True)
                writer.add_image('sample_image', grid, step)

                torch.save(generator.state_dict(), 'checkpoint_{}.pt'.format(step))

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
    opt = get_opt()
    train(opt)