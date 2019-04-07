# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 20:51:27 2018

@author: lenovo
"""
#针对双硬币模型
import numpy as np
from scipy import stats
observations = np.array([[1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
                         [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                         [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                         [1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
                         [0, 1, 1, 1, 0, 1, 1, 1, 0, 1]])
def em_single(initial,observations):
    """
    单次EM迭代
    """
    p_A=initial[0]
    p_B=initial[1]
    counts={'A':{'front':0,'reverse':0},'B':{'front':0,'reverse':0}}
    for observation in observations:
    #E步
        num=len(observation)
        front_num=sum(observation)
        reverse_num=num-front_num
        #A出现front_num次正面的可能性
        front_A=stats.binom.pmf(front_num,num,p_A)
        front_B=stats.binom.pmf(front_num,num,p_B)
        weight_A=front_A/(front_A+front_B)#计算的Q(Z)
        weight_B=front_B/(front_A+front_B)#计算的Q(Z)
        # 更新在当前参数下A、B硬币产生的正反面次数
        counts['A']['front']+=weight_A*front_num
        counts['A']['reverse']+=weight_A*reverse_num
        counts['B']['front']+=weight_B*front_num
        counts['B']['reverse']+=weight_B*reverse_num
    #M步
    new_p_A=counts['A']['front']/(counts['A']['front']+counts['A']['reverse'])
    new_p_B=counts['B']['front']/(counts['B']['front']+counts['B']['reverse'])
    return new_p_A,new_p_B

def EM(observations,initial,maxiter=10000,tol=1e-6):
    iter=0
    while(iter<maxiter):
        new_initial=em_single(initial,observations)
        initial_change=abs(initial[0]-new_initial[0])
        if initial_change<tol:
            break
        else:
            initial=new_initial
        iter+=1
    return [new_initial,iter]


if __name__ == "__main__":
    result=EM(observations,[0.65,0.54])
    print(result)

        
    
