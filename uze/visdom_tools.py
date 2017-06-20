import numpy as np
from visdom import Visdom
import datetime
import torch

class VisdomPlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name=None, port=8097):
        self.viz = Visdom(port=port)
        self.env = env_name
        if self.env is not None and self.env!='main':
            self.env = self.env + '-' + datetime.datetime.now().strftime('%b-%d-%y_%H-%M-%S')
        else:
            self.env = 'main'
        self.plots = {}

    def plot(self, title_name='Loss', xlabel_name='Epoch', ylabel_name='Loss', split_name=None, x=None, y=None, env=None):

        if env is not None:
            plot_env = env
        else:
            plot_env = self.env

        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=plot_env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel=xlabel_name,
                ylabel=ylabel_name
            ))
        else:
            self.viz.updateTrace(X=np.array([x]), Y=np.array([y]), env=plot_env, win=self.plots[title_name], name=split_name)

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )
    
    def imshow(self, img, title_name='Image', caption='Image Show', env=None):

        if env is not None:
            plot_env = env
        else:
            plot_env = self.env
        
        if isinstance(img, torch.autograd.Variable):
            img = img.data
        if isinstance(img, torch.Tensor):
            if img.is_cuda:
                img = img.cpu()
            img = img.numpy()

        self.viz.image(
                img,
                env = plot_env,
                opts=dict(
                title=title_name,
                caption = caption)
                )

    def imseqshow(self, imgs, title_name='Image Sequence', caption='Image Sequence Show', env=None):

        if env is not None:
            plot_env = env
        else:
            plot_env = self.env

        self.viz.images(
                imgs,
                env = plot_env,
                opts=dict(
                title=title_name,
                caption = caption)
                )
