import pandas as pd
import numpy as np
import math

class LaplacianNB():
    """
    关于二分类问题的Laplace平滑朴素贝叶斯
    """
    def __init__(self):
        """
        
        """
    def count_fun(self,list):
        """
        返回list的值和对应的频数
        """
        unique_freq={}
        for value in list:
            if value in unique_freq:
                unique_freq[value]+=1
            else:
                unique_freq[value]=1
        return unique_freq
   
    def discrete_prob(self,unique_freq,N_num):
        """
        拉普拉斯修正离散变量概率值
        """
        prob={}
        for value,freq in unique_freq.items():
            prob[value]=float(freq+1)/(N_num+len(unique_freq))
        return prob
    
    def mean_sd(self,list):
        """
        计算list的均值和标准差，计算连续属性的概率密度函数
        """
        mean=np.mean(list)
        sd=math.sqrt(np.var(list))
        return mean,sd
    
    def continous_prob(self,x,mean,sd):
        """
        概率密度函数，这里假设连续变量服从正态分布
        """
        p=1/(math.sqrt(2*math.pi)*sd)*math.exp(-(x-mean)**2/(2*sd**2))
        return p
    
    def train(self,x,y):
        """
        训练模型
        """
        N=len(y)
        self.classes=self.count_fun(y)
        self.classes_prob={}
        for value,freq in self.classes.items():
            self.classes_prob[value]=float(freq+1)/(N+len(self.classes))
        self.discrete_attr_with_good_p = []
        self.discrete_attr_with_bad_p = []
        for i in range(6):
            attr_with_good=[]
            attr_with_bad=[]
            for j in range(N):
                if y[j]=="是":
                    attr_with_good.append(x[j][i])
                else:
                    attr_with_bad.append(x[j][i])
            attr_with_good_count=self.count_fun(attr_with_good)
            attr_with_bad_count=self.count_fun(attr_with_bad)
            self.discrete_attr_with_good_p.append(self.discrete_prob(attr_with_good_count,self.classes["是"]))
            self.discrete_attr_with_bad_p.append(self.discrete_prob(attr_with_bad_count,self.classes["否"]))
        self.good_means=[]  
        self.good_sds=[]
        self.bad_means=[]
        self.bad_sds=[]
        for i in range(2):
            attr_with_good=[]
            attr_with_bad=[]
            for j in range(N):
                if y[j]=="是":
                    attr_with_good.append(x[j][i+6])
                else:
                    attr_with_bad.append(x[j][i+6])
            good_mean,good_sd=self.mean_sd(attr_with_good)
            bad_mean,bad_sd=self.mean_sd(attr_with_bad)
            self.good_means.append(good_mean)
            self.good_sds.append(good_sd)
            self.bad_means.append(bad_mean)
            self.bad_sds.append(bad_sd)
    def predict(self,new_x):
        p_good=self.classes_prob["是"]
        p_bad=self.classes_prob["否"]
        for i in range(6):
            p_good*=self.discrete_attr_with_good_p[i][new_x[i]]
            p_bad*=self.discrete_attr_with_bad_p[i][new_x[i]]
        for i in range(2):
            p_good*=self.continous_prob(new_x[i+6],self.good_means[i],self.good_sds[i])
            p_bad*=self.continous_prob(new_x[i+6],self.bad_means[i],self.bad_sds[i])
        if p_good>=p_bad:
            return p_good,p_bad,"是"
        else:
            return p_good,p_bad,"否"

if __name__=="__main__":
    LNB=LaplacianNB()
    data=pd.read_csv("C:/Users/lenovo/Desktop/r小代码/machine-learning/table4_3.csv",engine="python",encoding="gb18030")
    x=data.values[:,0:8]
    y=data.values[:,8]
    LNB.train(x,y)
    label = LNB.predict(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460])
    print ("predict ressult: ", label)
    