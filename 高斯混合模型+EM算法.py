# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 15:30:17 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt

def prob(x,mu,sigma):
    """
    多元高斯分布的概率密度函数
    """
    n=x.shape[1]
    sigma=np.mat(sigma)
    return math.exp(-0.5*(x-mu)*sigma.I*(x-mu).T)/(pow(2*math.pi,n)*(np.linalg.det(sigma)**0.5))

def EM(x,k,maxiter=1000,tol=1e-5):
    """
    EM算法
    k为高斯混合成分个数
    """
    m,n=x.shape
    #1.初始化模型参数，包括alpha,mu,sigma
    alpha=[1/k for i in range(k)]#1.1初始化高斯混合成分
    mu=[x[random.randint(0,m),] for i in range(k)]#1.2初始化mu，采用随机抽样k次
    sigma_k=[[0 for i in range(n)] for j in range(n)]#1.3初始化sigma，需要考虑的参数为变量个数和高斯混合成分个数
    sigma_k=np.mat(sigma_k)
    sigma_k=sigma_k.astype(float)
    for i in range(n):
        for j in range(n):
            if i==j:
                sigma_k[i,j]=float(0.1)
    sigma=[sigma_k for r in range(k)]
    gama=np.mat(np.zeros((m,k)))
    iter=0
    while(iter<maxiter):
        #E步求出各混合成分的后验概率gama
        gama_old=copy.deepcopy(gama)
        for i in range(m):
            gama_i_sum=0
            for j in range(k):
                gama[i,j]=alpha[j]*prob(x[i,],mu[j],sigma[j])
                gama_i_sum+=gama[i,j]
            for j in range(k):
                gama[i,j]/=gama_i_sum
        #M步更新mu，sigma，alpha
        gama_k_sum=np.sum(gama,axis=0)[0]
        for i in range(k):
            mu[i]=np.zeros((1,n))
            sigma[i]=np.zeros((n,n))
            for j in range(m):
                mu[i]+=gama[j,i]*x[j,]/gama_k_sum[0,i]
            for j in range(m):
                sigma[i]+=gama[j,i]*(x[j,]-mu[i]).T*(x[j,]-mu[i])/gama_k_sum[0,i]
            alpha[i]=gama_k_sum[0,i]/m
        if np.sum(abs(gama[1]-gama_old[1]))<tol*k:
           break            
        iter+=1
    return gama
        
                
def draw_cluster(x,k):
    gama=EM(x,k)
    m,n=x.shape
    cluster=[0 for i in range(m)]
    for i in range(m):
        cluster[i]=np.argmax(gama[i])   
    if n!=2:
        print("Sorry! I can not draw because the dimension of your data is not 2!")
    else:
        for i in range(m):
            colors=['b','g','r','orange']
            plot_x1=x[i,0]
            plot_x2=x[i,1]
            c=colors[cluster[i]]
            plt.scatter(plot_x1,plot_x2,c=c)
        plt.show()
if __name__=="__main__":
    x=pd.read_csv("C:/Users/lenovo/Desktop/r小代码/machine-learning/watermelon.csv",engine="python")
    x=x.astype(float)
    x=np.mat(x)
    draw_cluster(x,3)
    
    
            
    
