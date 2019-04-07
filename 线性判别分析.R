data<-iris
data$Species<-ifelse(data$Species=="setosa",1,
                     ifelse(data$Species=="versicolor",2,3))

library(MASS)
fit<-lda(Species~.,data=data)
fit
fit[1:length(fit)]

lda_fun<-function(data,new_sample){
  m<-ncol(data)
  n<-nrow(data)
  y<-data[,m]
  x<-data[,-m]
  x_col<-ncol(x)
  #计算各属性的均值
  u<-apply(x,2,mean)
  #计算类间散度矩阵，m*m的对称矩阵
  mean_fun<-function(value) {by(value,data[,m],mean)}#对每个属性按y的分组求均值
  mean_attr<-apply(x,2,mean_fun)
  matrix_between<-matrix(0,nrow=x_col,ncol=x_col)
  for(i in 1:length(unique(y))){
    number<-sum(y==unique(y)[i])
    matrix_between<-matrix_between+(mean_attr[i,]-u)%*%t(mean_attr[i,]-u)*number
  }
  tr_between<-qr(matrix_between)$rank
  #计算类内散度矩阵，m*m的对称矩阵
  matrix_within<-matrix(0,nrow=x_col,ncol=x_col)
  for(i in 1:length(unique(y))){
    x_data<-data[data$Species==unique(y)[i],-m]
    Swi<-matrix(0,nrow=x_col,ncol=x_col)
    for(j in 1:nrow(x_data)){
      Swi<-Swi+t(as.matrix(x_data[j,]-mean_attr[i,]))%*%as.matrix(x_data[j,]-mean_attr[i,])
    }
    matrix_within<-matrix_within+Swi
  }
  tr_within<-qr(matrix_within)$rank
  #求Sb/Sw的特征根和特征向量
  eigen_matrix<-solve(matrix_within)%*%matrix_between
  eigen_vector<-eigen(eigen_matrix)$vectors
  eigen_value<-eigen(eigen_matrix)$values
  #特征向量的个数=tr_within-tr_between,可以看到特征值里面有两个接近于0。由于浮点数的波动，
  #实际这两个特征值应该为0
  vector_num<-tr_within-tr_between
  #投影矩阵w
  w<-eigen_vector[,1:vector_num]
  #投影后的x
  x_project<-as.matrix(x)%*%w
  #以图像表示出来--二维
  if(ncol(x_project)==2){
  final<-as.data.frame(cbind(x_project,y))
  library(ggplot2)
  p<-ggplot(data=final)+
    geom_point(aes(x=V1,y=V2,color=as.factor(y)))
  print(p)
  }
  #查看新样本属于哪一个类别
  new_sample<-matrix(new_sample,nrow=1)
  N<-length(unique(y))
  dist<-vector(length=N)
  for(i in 1:N){
   dist[i]<- ((new_sample-mean_attr[i,])%*%w)%*%t((new_sample-mean_attr[i,])%*%w)
  }
  return(which.min(dist))
}

lda_fun(data,new_sample = c(0.3,1.5,2.5,3.5))
