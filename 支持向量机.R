data<-read.csv("C:\\Users\\lenovo\\Desktop\\r小代码\\machine-learning\\watermelon.csv")
data<-as.matrix(data)
datamat<-data[,1:2]
labelmat<-as.matrix(data[,3],ncol=1)
#在0~m中选择一个不是i的整数
selectJrand_fun<-function(i,m){
  j<-round(runif(1,0,m),0)
  while(i==j){
    j<-round(runif(1,0,m),0)
  }
  return(j)
}
#保证L<=a<=H
clipAlpha_fun<-function(a,L,H){
  if(a>H){
    a<-H
  }else if(a<L){
    a<-L
  }
  return(a)
}
#-----------------------------------------step1 核函数-------------------------------------------
#x为支持向量特征，row为某一行特征数据，kernelClass包括核函数类型(线性核linear、高斯核gaussian为代表)和参数

kernelTrans_fun<-function(x,row,kernelclass){
  m<-nrow(x)
  n<-ncol(x)
  row<-matrix(row,nrow=1)
  k<-matrix(nrow=m,ncol=1)#核函数结果,m*m维对称矩阵
  if(kernelclass[1]=="linear"){
    k<-x%*%t(row)
  }else if(kernelclass[1]=="gaussian"){
    for(i in 1:m){
      k[i,1]<-exp(-(x[i,]-row)%*%t(x[i,]-row)/as.numeric(kernelclass[2])^2*0.5)
    }
  }
  return(k)
}
#-----------------------------------step2 定义一个list方便存储数据--------------------------
save_list_fun<-function(datamat,labelmat,c,tol,kernelclass){
  save_list<-list()
  save_list$x<-datamat#特征值矩阵
  save_list$label_mat<-labelmat#分类
  save_list$c<-c#软阈值参数
  save_list$tol<-tol#容错大小
  m<-nrow(datamat)
  save_list$m<-m
  save_list$alpha<-matrix(0,nrow=m)#拉格朗日乘子alpha初始值
  save_list$b<-0#截距项初始值
  save_list$error_cache<-matrix(0,nrow=m,ncol=2)#缓存误差
  save_list$k<-matrix(nrow=m,ncol=m)#核函数结果
  for(i in 1:m){
   save_list$k[,i]<- kernelTrans_fun(datamat,datamat[i,],kernelclass)
  }
  return(save_list)
}
#计算函数对xi的预测值与yi的差值
Error_fun<-function(save_list,row_num){
  y_hat<-t(as.vector(save_list$alpha)*save_list$label_mat)%*%save_list$k[,row_num]
return(y_hat-save_list$label_mat[row_num])
}
#选择优化的第二个对偶因子
second_dual_fun<-function(save_list,i,error_i){#i&error_i为选取的第一个alpha对象号和对象差值
  j<-0
  max_delta_error<-0  
  error_j<-0
  save_list$error_cache[i,]<-c(1,error_i)
  being_selected_j<-which(save_list$error_cache[,1]==1)
  if(length(being_selected_j)>1){
    for(k in 1:length(being_selected_j)){
      if(k==i){
        break
      }
      error_k<-Error_fun(save_list,k)
      delta_error<-abs(error_k-error_i)
      if(delta_error>max_error){
        max_delta_error<-delta_error
        error_j<-error_k
        j<-k
      }
    }
  } else {
    j<-selectJrand_fun(i,save_list$m)
    error_j<-Error_fun(save_list,j)
  }
  result<-list(j,error_j)
  names(result)<-c("j","error_j")
  return(result)
}
#更新error_cache第i行值
update_error_fun<-function(save_list,i){
  error_i<-Error_fun(save_list,i)
  save_list$error_cache[i,]<-c(1,error_i)
  return(save_list)
}
#---------------------------step3 对对偶因子向量的单个元素求解并更新截距项--------------------
inner_fun<-function(save_list,i){
  error_i<-Error_fun(save_list,i)
#首先检查alpha_i是否满足KKT条件，如果不满足则找到alpha_j，要求|error_i-error_j|达到最大，这样alpha_j的变化最大
 if((save_list$label_mat[i]*error_i<(-save_list$tol)&save_list$alpha[i]<save_list$c)|
    (save_list$label_mat[i]*error_i>save_list$tol&save_list$alpha[i]<save_list$c)){
   j<-second_dual_fun(save_list,i,error_i)$j
   error_j<-second_dual_fun(save_list,i,error_i)$error_j
   # 根据对象 i 、j 的类标号（相等或不等）确定KKT条件的上界和下界
   if(save_list$label_mat[i]!=save_list$label_mat[j]){
     L<-max(0,save_list$alpha[j]-save_list$alpha[i])
     H<-min(save_list$c,save_list$c+save_list$alpha[j]-save_list$alpha[i])
   }else{
     L<-max(0,save_list$alpha[j]+save_list$alpha[i]-save_list$c)
     H<-min(save_list$c,save_list$alpha[j]+save_list$alpha[i]) 
   }
   if(L==H){
    # print("L==H")
     return(0)
   }
   eta<-save_list$k[i,i]+save_list$k[j,j]-save_list$k[i,j]*2
   if(eta<=0){
     print("eta==0")
     return(0)
   }
   #优化第二个alpha
   alpha_i_old<-save_list$alpha[i]
   alpha_j_old<-save_list$alpha[j]
   save_list$alpha[j]<-save_list$alpha[j]+save_list$label_mat[j]*(error_i-error_j)/eta
   save_list$alpha[j]<-clipAlpha_fun(save_list$alpha[j],L,H)
   save_list<-update_error_fun(save_list,j)#更新差值矩阵
   if(abs(save_list$alpha[j]-alpha_j_old)<0.0001){
     #print("j is not moving enough")
     return(0)
   }
   save_list$alpha[i]<-alpha_i_old+save_list$label_mat[i]*save_list$label_mat[j]*(alpha_j_old-save_list$alpha[j])
   save_list<-update_error_fun(save_list,i)#更新差值矩阵
   #计算截距b
   b1_new<--error_i-save_list$label_mat[i]*save_list$k[i,i]*(save_list$alpha[i]-alpha_i_old)-
     save_list$label_mat[j]*save_list$k[j,i]*(save_list$alpha[j]-alpha_j_old)+save_list$b
   b2_new<--error_j-save_list$label_mat[i]*save_list$k[i,j]*(save_list$alpha[i]-alpha_i_old)-
     save_list$label_mat[j]*save_list$k[j,j]*(save_list$alpha[j]-alpha_j_old)+save_list$b
  if((0<save_list$alpha[i])&(save_list$alpha[i]<save_list$c)){
    save_list$b<-b1_new
  }else if((0<save_list$alpha[j])&(save_list$alpha[j]<save_list$c)){
    save_list$b<-b2_new
  }else{
    save_list$b<-(b1_new+b2_new)/2
  }
   return(1)
 }else{#满足KKT条件
    return(0)
  }
}
#-------------------------------------step4 SMO算法求解对偶因子------------------------------------
SMO_fun<-function(datamat,labelmat,c,tol,kenerclass="linear",maxiter){
  save_list<-save_list_fun(datamat,labelmat,c,tol,kenerclass)
  iter<-0
  entireset<-TRUE#是否遍历整个数据集
  alphapairschange<-0#迭代优化的次数
  # 从选择第一个 alpha 开始，优化所有alpha
  while(iter<maxiter&(alphapairschange>0|entireset)){#遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
    alphapairschange<-0
    if(entireset){#遍历整个数据集
      for(i in 1:save_list$m){
        alphapairschange<-alphapairschange+inner_fun(save_list,i)
      }
      sprintf("全样本遍历:第%i次迭代 样本%i,alpha优化次数%f",iter,i,alphapairschange)
      iter<-iter+1
    }else{#遍历非边界点
      bound<-which(save_list$alpha<save_list$c&save_list$alpha>0)
      for(i in 1:length(bound)){
        alphapairschange<-alphapairschange+inner_fun(save_list,i) 
        sprintf("非边界点遍历:第%i次迭代 样本%i,alpha优化次数%f",iter,i,alphapairschange)
        iter<-iter+1
      }
      if(entireset==TRUE){
        entireset==FALSE#遍历一次全样本便遍历非边界点
      }
      if(alphapairschange==0){
        entireset==TRUE#如果alpha没有更新,计算全样本遍历
      }
    }
    sprintf("迭代次数: %i", iter)
  }
  result<-list(save_list$alpha,save_list$b)
  names(result)<-c("alpha","b")
  return(result)
}
