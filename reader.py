import random, struct, sys, os
import numpy as np
import torch
from torch.utils.data import Dataset


class LorenzDataset(Dataset):
    """
    """
    def __init__(self, args, data_dir, res=1, train=True,
        valid=False, train_valid_split=0.1, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.args = args
        self.data_dir = data_dir
        self.res = res
        self.train = train
        self.valid = valid
        self.train_valid_split = train_valid_split

        # Generate seed
        self.N = int(1e2)
        self.s0 = np.random.rand(self.N, 3)
        self._dt = 0.01
        
    @property
    def dt(self):
        return self._dt


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        i = random.randrange(0, self.N)

        total_time = self.args.input_len + self.args.output_len

        # Generate data on-the-fly
        states = self.gen_lorenz_series(self.s0[i], total_time, 1)
        #states = self.load_data_from_file(i)

        data = states[:self.args.input_len,]
        label = states[self.args.input_len:,]

        data = torch.from_numpy(data).type(torch.FloatTensor)
        label = torch.from_numpy(label).type(torch.FloatTensor)

        return data, label

    def lorenz(self, x, y, z, s=10, r=28, b=2.667):
        x_dot = s*(y - x)
        y_dot = r*x - y - x*z
        z_dot = x*y - b*z
        return np.array([x_dot, y_dot, z_dot])

    def gen_lorenz_series(self, s0, num_steps, num_freq):
        # dt = 0.01

        s = np.empty((num_steps,3))
        s[0] = s0
        ss = np.empty((num_steps//num_freq,3))
        j = 0
        for i in range(num_steps-1):
            # Derivatives of the X, Y, Z state,
            if i%num_freq ==0:
                ss[j] = s[i]
                j += 1
            sdot = self.lorenz(s[i,0], s[i,1], s[i,2])
            s[i + 1] = s[i] + sdot  * self._dt

        return ss


