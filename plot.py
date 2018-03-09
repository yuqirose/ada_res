import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10


def plot_line3d(ax, dat):
    """"""
    n = dat.shape[0] 
    x, y, z = dat[:,0], dat[:,1], dat[:,2]

    ax.plot(x, y, z, label='parametric curve')
    ax.legend()


def plot_scatter3d(ax, dat, color):
    """   """
    n = dat.shape[0]
    xs, ys, zs = dat[:,0], dat[:,1], dat[:,2]

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
  
    ax.scatter(xs, ys, zs, c=color, marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')


def plot_seq2seq(data, target):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    bat_sz = data.shape[0]
    plot_scatter3d(ax,  data[0,:], 'k')
    plot_scatter3d(ax,  target[0,:], 'r')
    # plt.show()


