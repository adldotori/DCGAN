import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),             
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Tanh()
        ) 
        self.apply(_weights_init)

    def forward(self, z):
        z = z.view(z.size(0), 100, 1, 1)
        ret = self.model(z)
        return ret.view(z.size(0), -1, 64, 64)
         
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = x.view(x.size(0), 1, 64, 64)
        ret = self.model(x)
        ret = ret.view(ret.size(0), 1)
        return ret

if __name__ == '__main__':
    batch_size = 3

    generator = Generator()
    generator.cuda()
    noise = torch.rand(batch_size, 100).cuda()
    gen = generator(noise)
    print(gen.shape)

    discriminator = Discriminator()
    discriminator.cuda()
    image = torch.rand(batch_size, 1, 64, 64).cuda()
    dis = discriminator(image)
    print(dis.shape)