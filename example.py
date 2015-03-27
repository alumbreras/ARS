# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:58:07 2015

@author: Alberto Lumbreras
"""
from __future__ import  division
import numpy as np
from matplotlib import pyplot as plt

from  ars import ARS


######################################
# Example 1: sample 10000 values 
# from the normal distribution N(2,3)
######################################
def f(x, mu=0, sigma=1):
    """ 
    Log Normal distribution 
    """
    return -1/(2*sigma**2)*(x-mu)**2
    
def fprima(x, mu=0, sigma=1):
    """
    Derivative of Log Normal distribution
    """
    return -1/sigma**2*(x-mu)

x = np.linspace(-100,100,100)
ars = ARS(f, fprima, xi = [-4,1,40], mu=2, sigma=3)
samples = ars.draw(10000)
plt.hist(samples, bins=100, normed=True)
plt.show()


######################################
# Example 2: sample 10000 values 
# from a gamma(2,0.5)
######################################
def f(x, shape, scale=1):
    """ 
    Log gamma distribution 
    """
    return (shape-1)*np.log(x)-x/scale
    
    
def fprima(x, shape, scale=1):
    """
    Derivative of Log gamma distribution
    """
    return (shape-1)/x-1/scale

x = np.linspace(-100,100,100)
ars = ARS(f, fprima, xi = [0.1,1,40], lb=0, shape=2, scale=0.5)
samples = ars.draw(10000)
plt.hist(samples, bins=100, normed=True)
plt.show()


######################################
# Example 3: sample 10000 values 
# from a beta(1.3,2.7) distribution
######################################
def f(x, a=1.3, b=2.7):
    """ 
    Log beta distribution 
    """
    return (a-1)*np.log(x)+(b-1)*np.log(1-x)
    
    
def fprima(x, a=1.3, b=2.7):
    """
    Derivative of Log beta distribution
    """
    return (a-1)/x-(b-1)/(1-x)

x = np.linspace(-100,100,100)
ars = ARS(f, fprima, xi = [0.1, 0.6], lb=0, ub=1, a=1.3, b=2.7)
samples = ars.draw(10000)
plt.hist(samples, bins=100, normed=True)
plt.show()
