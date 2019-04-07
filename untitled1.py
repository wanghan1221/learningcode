# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 15:42:47 2018

@author: lenovo
"""

import pandas as pd
import numpy as np
import math
data=pd.read_csv('C:/Users/lenovo/Desktop/r小代码/machine-learning/table3_0.csv',engine='python')
data['好瓜']=data.好瓜.apply(lambda x: 1 if x=='是'  else 0 )

def split_discrete_fun(data,var_index,value):
    '''
    data是dataframe格式
    var_index是变量列号
    value是变量对应的值
    该函数按照value进行分割
    '''
    returndata=[]
    for d in data.values:
        if d[var_index]==value:
            returndata.append(d)
    returndata=pd.DataFrame(returndata)
    return returndata

def label_predict(data):
    '''
    单个特征值，当前结点的类别设定为该节点样本最多的类别
    '''
    labels=data.values[:,-1].tolist()
    labels_count={}
    for label in labels:
        if label not in labels_count.keys():
            labels_count[label]=0
        labels_count[label]+=1
    return max(labels_count,key=labels_count.get)
    
    
def decisionstump(data,D):
    '''
    单层决策树
    基于当前样本权值分布D计算错分率，错分率最小的特征值建立的单层决策树为最佳决策树
    weight_error是Adaboost和分类器交互的地方
    '''
    m,n=data.shape
    names=data.columns.values.tolist()
    min_weighterror=float('inf')
    best_predict=[]
    best_stump={}
    for i in range(n-1):
        predict={}
        values=set(data[names[i]])
        for value in values:
            split_data=split_discrete_fun(data,i,value)
            predict[value]=label_predict(split_data)
        predict_result=[]
        for j in range(m):
            predict_result.append(predict[data.values[j,i]])   
        error=np.mat(np.ones((m,1)))#存储预测结果与实际值是否相等，0表示相等，1表示不等
        for j in range(m):
            if predict_result[j]==data.values[j,-1]:
                error[j,]=0
            else:
                error[j,]=1
        weight_error=D.T*error
        if weight_error<min_weighterror:
            min_weighterror=weight_error
            best_predict=predict_result
            best_stump['var_index']=i
    return min_weighterror,best_predict,best_stump

def Adaboost(data,T):
    '''
    T为分类器个数，也是迭代次数
    '''
    m,n=data.shape
    D=np.mat(np.ones((m,1))/m)
    labels=np.mat(data.values[:,-1]).T
    Hx=np.mat(np.zeros((m,1)))
    weakclassifier=[]
    for t in range(T):
        min_weighterror,best_predict,best_stump=decisionstump(data,D)
        best_predict=np.mat(best_predict).T
        alpha=0.5*np.log((1-min_weighterror)/max(min_weighterror,1e-16))#防止error为0不会发生除0溢出
        best_stump['alpha']=alpha
        weakclassifier.append(best_stump)
        D_new=np.multiply(-alpha*labels.T,best_predict)
        D_new=np.exp(D_new.astype('float'))*D
        D_new/=sum(D_new)
        D=D_new
        Hx+=best_predict*alpha
        final_predict=np.sign(Hx) 
        final_error=(final_predict!=labels).T*np.ones((m,1))/m
        if final_error==0:
            break
    return weakclassifier,final_error
        
        
            
        
            
        
    

    
            

        
        
        
        
        
    
