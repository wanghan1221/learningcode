library(tictoc)
x1<-c(0,0,1,1)
x2<-c(0,1,0,1)
y<-c(0,1,1,0)
data<-cbind(x1,x2,y)
#隐层个数q（大于输入层个数），输出层个数l
RBF_fun<-function(data,q,l,rate){
  #输入层个数
  d<-ncol(data)-1
  n<-nrow(data)
  x<-data[,1:d]
  y<-as.matrix(data[,d+1],nrow=n)
  #step1:确定隐层神经元中心，可随机采样、聚类。采用随机
  c<-matrix(runif(q),nrow=1)
  #step2:确定参数w和beta：w为隐层神经元对应的权重，beta为高斯径向函数对应的权重
  #先随机设置初始值
  w<-matrix(runif(q),nrow=1)
  beta<-matrix(runif(q),nrow=1)
  #再利用bp算法迭代确定w和beta的值
  iter<-0#迭代次数
  old_error<-0#前一次迭代的累积误差
  same_num<-0#同样的累计误差累积次数
  while(TRUE){
    iter<-iter+1
    #计算径向函数值
    p<-matrix(nrow = n,ncol=q)
    for(i in 1:n){
      for(j in 1:q){
        p[i,j]<-exp(-beta[1,j]*matrix((x[i,]-c[1,j]),nrow=1)%*%matrix((x[i,]-c[1,j]),ncol=1))
      }
    }
    y_hat<-p%*%t(w)
    new_error<-t(y_hat-y)%*%(y_hat-y)/2
    #更新w和beta
    d_w<-t(y_hat-y)%*%p
    d_beta<-matrix(0,nrow=1,ncol=q)
    for(i in 1:q){
      d_beta[i]<--t(y_hat-y)%*%(x-c[,i])%*%t(x-c[,i])%*%p[,i]%*%w[1,i]
    }
    w<-w-rate*d_w
    beta<-beta-rate*d_beta

  if(abs(new_error-old_error)<0.001){
    same_num<-same_num+1
    if(same_num==100){
      break
    }
  }else{
    same_num<-0
    old_error<-new_error
  }
 }
  return(list(iter,y_hat,new_error,same_num))
}
RBF_fun(data,q=10,l=1,rate=0.1)
