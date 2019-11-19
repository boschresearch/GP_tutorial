"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
@author: Barbara Rakitsch
"""

import matplotlib.pylab as plt
import numpy as np
from scipy.stats import multivariate_normal, norm

def beautify_plot(ax, xlabel=None, ylabel=None):
    """
    Args:
        ax       :    axis of subplot
        xlabel   :    xlabel
        ylabel   :    ylabel
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)


def contours(mu, Sigma):
    """
    Args:
        mu      :   mean of Gaussian distribution
        Sigma   :   covariance matrix of Gaussian distribution

    Returns:
        arr     :   Gaussian probabilities on a grid of size [-3,3] with 100 steps
    """
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1, x2)
    x = np.column_stack([x1.flat, x2.flat])
    z = multivariate_normal.pdf(x, mu, Sigma).reshape(x1.shape)
    return x1, x2, z
