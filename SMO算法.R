data<-read.csv("C:\\Users\\lenovo\\Desktop\\r小代码\\machine-learning\\watermelon.csv")
data<-data[,-3]
#n样本量，m输出神经元个数，d输入神经元个数
#-------------------------step1每一个样本都有一个权重矩阵，m*d维度------------------------------
init_weight_fun<-function(n,m,d){
  weight<-array(runif(n*m*d),dim=c(m,d,n))
  return(weight)
}
weight<-init_weight_fun(16,2,2)
#计算向量的二范数
cal2NF_fun<-function(x){
  res<-0
  for(i in 1:length(x)){
    res<-res+x[i]^2
  }
  return(res^0.5)
}
#对数据进行归一化处理
normalize_fun<-function(data){
  newdata<-data
  for(i in 1:nrow(data)){
    for(j in 1:length(data[i,])){
      newdata[i,j]<-data[i,j]/cal2NF_fun(data[i,])
    }
  }
  result<-list(newdata,data)
  names(result)<-c("newdata","olddata")
  return(result)
}
#对权重数组进行归一化
normalize_weight_fun<-function(weight){
  newweight<-weight
  for(i in 1:dim(weight)[3]){
    newweight[,,i]<-normalize_fun(weight[,,i])$newdata
  }
  return(newweight)
}
#---------------------------------step2 得到最佳匹配单元索引值-------------------------------
getwinner_fun<-function(sample,weight){
  max_dist<-0
  for(i in 1:dim(weight)[3]){
    for(j in 1:dim(weight)[1]){
    #因为data和weight都进行了归一化，所以求样本与权向量的距离最小值可以转换为求max(Wj,Xi)
    if(sum(as.matrix(sample,nrow=1)%*%weight[j,,i])>max_dist){
      max_dist<-sum(as.matrix(sample,nrow=1)%*%weight[j,,i])
      mark_n<-i
      mark_m<-j
    }
    }
  }
  return(c(mark_n,mark_m))
}
#---------------------------------------step3 得到神经元相邻领域-------------------------------
#n和m是获胜神经元下标，由getwinner_fun得到，radius为邻域半径
getneighbor_fun<-function(n,m,radius,weight){
  neighbor<-data.frame()
  for(i in 1:dim(weight)[3]){
    for(j in 1:dim(weight)[1]){
      N<-((i-n)^2+(j-m)^2)^0.5
      if(N<radius){
        this_one<-c(i,j,N)
        neighbor<-rbind(neighbor,this_one)
        names(neighbor)<-c("n","m","N")
      }
    }
  }
  return(neighbor)
}
#-----------------------------------------step4 学习率--------------------------------------
#ω(t+1)= ω(t)+ η(t，n) * (x-ω(t))
#η(t，n):η为学习率是关于训练时间t和与获胜神经元的拓扑距离n的函数。
#η(t，n)=η(t)e^(-n)
#η(t)一般取迭代次数的倒数
rate_fun<-function(t,N){
  rate<-1/t*exp(-N)#0.3/(t+1)*exp(-N)
  return(rate)
}
#----------------------------------------step5 SMO算法------------------------------------
SMO_fun<-function(data,weight,radius,maxiter){
  iter<-0
  while(iter<maxiter){
    iter<-iter+1
    weight<-normalize_weight_fun(weight)
    for(i in 1:nrow(data)){
      sample<-data[i,]
      n<-getwinner_fun(sample,weight)[[1]]
      m<-getwinner_fun(sample,weight)[[2]]
      neighbor<-getneighbor_fun(n,m,radius,weight)
      for(j in 1:nrow(neighbor)){
        a1<-neighbor[j,1]
        a2<-neighbor[j,2]
        a3<-neighbor[j,3]
        rate<-rate_fun(iter,a3)
        #调节权重
        #r里面对array的处理挺麻烦的，一不小心就变成了list
        for(k in 1:dim(weight)[2]){
          weight[a2,k,a1]<-weight[a2,k,a1]+rate*(sample[1,k]-weight[a2,k,a1]) 
        }
      }
      radius<-radius*exp(1/iter)
    }
  }
  N<-nrow(data)
  M<-dim(weight)[1]
  key<-NULL
  for(p in 1:nrow(data)){
    final_n<-getwinner_fun(data[p,],weight)[1]
    final_m<-getwinner_fun(data[p,],weight)[2]
    key[p]<-final_n*M+m
  }
  id<-1:nrow(data)
  group<-rep(0,length=nrow(data))
  d<-cbind(id,key,group)
  a<-unique(key)
 for(g in 1:length(a)){
   d[which(d[,2]==a[g]),3]<-g
 }
  plot_data<-cbind(data,d[,3])
  names(plot_data)[3]<-"group"
  library(ggplot2)
  p<-ggplot(plot_data)+geom_point(aes(density,sweet,color=as.factor(group)))
  print(p)
}
SMO_fun(data,weight,3,100)
