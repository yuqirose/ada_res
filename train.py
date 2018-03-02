import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt 

"""adaptive resolution sequence prediction"""


def train(train_loader, epoch, model, args):
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

	train_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		
		#transforming data
		#data = Variable(data)
		#to remove eventually, data size 28 (step) x 128 (batch) x 28 (dim)
		data = Variable(data.squeeze().transpose(0, 1))
		data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])
		
		#forward + backward + optimize
		optimizer.zero_grad()
		output = model(data)
        loss = F.mse_loss(output, target)
		loss.backward()
		optimizer.step()

		#grad norm clipping, only in pytorch version >= 1.10
		nn.utils.clip_grad_norm(model.parameters(), clip)

		#printing
		if batch_idx % print_every == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader),
				kld_loss.data[0] / args.batch_size,
				nll_loss.data[0] / args.batch_size))

			sample = model.sample(28)
			plt.imshow(sample.numpy())
			plt.pause(1e-6)

		train_loss += loss.data[0]


	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, train_loss / len(train_loader.dataset)))


def test(test_loader, epoch):
	"""uses test data to evaluate 
	likelihood of the model"""
	
	loss = 0.0
	for i, (data, target) in enumerate(test_loader):                                            
		
		#data = Variable(data)
		data = Variable(data.squeeze().transpose(0, 1))
		data = (data - data.min().data[0]) / (data.max().data[0] - data.min().data[0])

		output = model(data)
		loss = F.mse_loss(output, target)
		loss /= len(test_loader.dataset)

	print('====> Test set loss: Loss = {:.4f} '.format(
		loss))
