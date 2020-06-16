from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

class Dataset(data.Dataset):
    def __init__(self, name, mode=None):
        super(Dataset, self).__init__()

        if name == 'mnist':
            self.dataset = datasets.MNIST('data/MNIST/', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(0.5,))
                        ]))
        elif mode == 'colab':
            self.dataset = datasets.ImageFolder('/content/gdrive/My Drive/celeba/', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ]))

        else:
            self.dataset = datasets.CelebA('data/CelebA/', download=True,
                        transform=transforms.Compose([
                            transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                        ]))

    def name(self):
        return "Dataset"

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

class Dataloader(object):
    def __init__(self, opt, dataset):
        super(Dataloader, self).__init__()
        use_cuda = not torch.cuda.is_available()
        kwargs = {'num_workers': opt.num_workers} if use_cuda else {}

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)

        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num-workers', type=int, default = 4)
    parser.add_argument('-e', '--epoch', type=int, default=40)
    parser.add_argument('-b', '--batch-size', type=int, default = 100)
    parser.add_argument('-d', '--display-step', type=int, default = 600)
    parser.add_argument('--dataset', type=str, default = 'mnist', help='mnist or celeba')
    opt = parser.parse_args()

    dataset = Dataset(opt.dataset)
    data_loader = Dataloader(opt, dataset)

    print('[+] Size of the dataset: %05d, dataloader: %03d' \
        % (len(dataset), len(data_loader.data_loader))) 