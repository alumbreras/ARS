# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:58:07 2015

@author: LUMBRERASA
"""
from __future__ import  division
import numpy as np
from matplotlib import pyplot as plt

from  ars import ARS

g = lambda x: np.exp(-x**2)
h = lambda x: -x**2
h_prima = lambda x: -2*x

def logpdf(xt,pdfargs=None):
    """
    Returns h(x) and h'(x)
    """
    return h(xt), h_prima(xt)

def f(x, mu=0, sigma=1):
    """ 
    Normal distribution 
    """
    return -1/(2*sigma**2)*(x-mu)**2
    
def fprima(x, mu=0, sigma=1):
    """
    Derivative of Normal distribution
    """
    return -1/sigma**2*(x-mu)


x = np.linspace(-100,100,100)

ars = ARS(f, fprima, np.array([-4,1,40]), mu=2, sigma=5)
samples = ars.draw(10000)
plt.hist(samples, bins=100, normed=True)
plt.show()
