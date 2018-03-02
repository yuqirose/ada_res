
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.nn import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim
from train import train, test
import matplotlib.pyplot as plt
from seq2seq import Seq2Seq


# Training settings
parser = argparse.ArgumentParser(description='adaptive vrnn')
parser.add_argument('--exp', type=int, default=64, metavar='N')
parser.add_argument('--sess', default="", metavar='N')

parser.add_argument('--data-dir', type=str, default="./data/data.npy", metavar='N')
parser.add_argument('--log-dir', default="/tmp/logs/multires", help='')
parser.add_argument('--model', type=str, default="vrnn", metavar='N')

parser.add_argument('--dataset', type=str, default="train", metavar='N')

parser.add_argument('--train-size', type=int, default=64, metavar='N')
parser.add_argument('--val-size', type=int, default=64, metavar='N')
parser.add_argument('--test-size', type=int, default=64, metavar='N')

parser.add_argument('--input-len', type=int, default=10, metavar='N')
parser.add_argument('--output-len', type=int, default=90, metavar='N')
parser.add_argument('--x-dim', type=int, default=3
    , metavar='N')
parser.add_argument('--h-dim', type=int, default=16, metavar='N')
parser.add_argument('--z-dim', type=int, default=8, metavar='N')

parser.add_argument('--batch-size', type=int, default=5, metavar='N')
parser.add_argument('--n-layers', type=int, default=1, metavar='N')
parser.add_argument('--n-epochs', type=int, default=10, metavar='N',
                                        help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                                        help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                        help='SGD momentum (default: 0.5)')
parser.add_argument('--l2', type=float, default=0.1, metavar='LR')
parser.add_argument('--opt', default="sgd")


parser.add_argument('--no-cuda', action='store_true', default=False,
                                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                                        help='random seed (default: 1)')
parser.add_argument('--vis-scalar-freq', type=int, default=10, metavar='N',
                                        help='how many batches to wait before logging training status')
parser.add_argument('--ckpt-freq', type=int, default=100, metavar='N',
                                        help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N')
parser.add_argument('--valid-freq', type=int, default=500, metavar='N')
parser.add_argument('--prev-ckpt', default="", help='')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()




if __name__ == "__main__":
    if args.cuda and torch.cuda.is_available(): print("Using CUDA")

    #hyperparameters
    x_dim = 28
    h_dim = 100
    z_dim = 16
    n_layers =  1
    n_epochs = 100
    clip = 10
    learning_rate = 1e-3
    batch_size = 128
    seed = 128
    print_every = 100
    save_every = 10

    #manual seed
    torch.manual_seed(seed)
    plt.ion()

    #init model + optimizer + datasets
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
            transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, 
            transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    model = Seq2Seq(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        
        #training + testing
        train(train_loader, epoch, optimizer)
        test(test_loader, epoch)

        #saving model
        if epoch % save_every == 1:
            fn = 'saves/vrnn_state_dict_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), fn)
            print('Saved model to '+fn)
