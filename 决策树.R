data<-read.csv("C:\\Users\\lenovo\\Desktop\\r小代码\\machine-learning\\table4.3.csv",stringsAsFactors=F)
#以最后一列为分类类别
#--------------------------------------step1 计算信息熵--------------------------------------
info_entropy<-function(data){
  if(is.null(dim(data))==FALSE){
  m<-length(data)
  n<-nrow(data)
  y<-data[,m]
  ent<-0
  for(i in 1:length(unique(y))){
    p<-length(y[y==unique(y)[i]])/n
    this_ent<-ifelse(p==0,0,-p*log2(p))
    ent<-ent+this_ent
  }
  }else{
   y<-data
   ent<-0
   for(i in 1:length(unique(y))){
     p<-length(y[y==unique(y)[i]])/length(data)
     this_ent<-ifelse(p==0,0,-p*log2(p))
     ent<-ent+this_ent
   }
 }
  return(ent)
}
#----------------------------------step2 按照给定特征划分数据集---------------------------------
#要求输入数据集，划分特征变量的序号，特征值(针对离散特征变量)
split_discrete_fun<-function(data,var_index,value){
  return_data<-data[data[,var_index]==value,-var_index]
  return(return_data)
}
#针对连续特征变量,direction决定是划分出大于value还是小于等于value的数据集
split_continuous_fun<-function(data,var_index,value,direction){
  if (direction==0) {
    return_data<-data[data[,var_index]<=value,-var_index]
  } else {
    return_data<-data[data[,var_index]>value,-var_index]
  }
  return(return_data)
}
#---------------------------------step3 基于信息增益选择最佳特征变量--------------------------
#从输入的训练样本集中，计算划分之前的熵，找到当前有多少个特征，遍历每一个特征计算信息增益，找到这些特征中能带来信息增益最大的那一个特征。 
#这里用分了两种情况，离散属性和连续属性 
#1、离散属性，在遍历特征时，遍历训练样本中该特征所出现过的所有离散值，假设有n种取值，那么对这n种我们分别计算每一种的熵，最后将这些熵加起来 
#就是划分之后的信息熵 
#2、连续属性，对于连续值就稍微麻烦一点，首先需要确定划分点，用二分的方法确定（连续值取值数-1）个切分点。遍历每种切分情况，对于每种切分， 
#计算新的信息熵，从而计算增益，找到最大的增益。 
#假设从所有离散和连续属性中已经找到了能带来最大增益的属性划分，这个时候是离散属性很好办，直接用原有训练集中的属性值作为划分的值就行，但是连续 
#属性我们只是得到了一个切分点，这是不够的，我们还需要对数据进行二值处理。
choose_feature_fun<-function(data){
  baseentropy<-info_entropy(data)
  bestinfogain<-0
  bestfeature<-0
  split_labels<-list()
  for(i in 1:(ncol(data)-1)){
    feature<-data[,i]
    #针对连续型特征变量，先对特征值进行排序，选取信息熵最小的划分值作为该特征变量划分值
    if(class(feature)=="numeric"){
      feature_sorted<-sort(feature)
      split_feature<-NULL
      for(j in 1:(length(feature_sorted)-1)){
        split_feature[j]<-(feature_sorted[j]+feature_sorted[j+1])/2
      }
      best_split_entropy<-1000
      for (value in split_feature){
        subset_data0<-split_continuous_fun(data,i,value,0)
        subset_data1<-split_continuous_fun(data,i,value,1)
        prob0<-nrow(subset_data0)/nrow(data)
        prob1<-nrow(subset_data1)/nrow(data)
        new_entropy<-info_entropy(subset_data0)*prob0+info_entropy(subset_data1)*prob1
        if(new_entropy<best_split_entropy){
          best_split_entropy<-new_entropy
          best_split<-value
        }
      }
      split_labels[i]<-best_split
      infogain<-baseentropy-best_split_entropy
    }else{
      #针对离散型
      new_entropy<-0
      for(j in unique(feature)){
        subset_data<-split_discrete_fun(data,i,j)
        if(is.null(dim(subset_data))==FALSE){
        this_entropy<-nrow(subset_data)/nrow(data)*info_entropy(subset_data)
        new_entropy<-new_entropy+this_entropy
        }else{
          this_entropy<-length(subset_data)/nrow(data)*info_entropy(subset_data)
        }
      }
      infogain<-baseentropy-new_entropy
    }
    if(infogain>bestinfogain){
      bestinfogain<-infogain
      bestfeature<-i
    }
  }
  #若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理  
  # 即是否小于等于bestSplitValue,例如将密度变为密度<=0.3815  
  #将属性变了之后，之前的那些float型的值也要相应变为0和1  
  if(class(data[,bestfeature])=="numeric"){
    bestsplitvalue<-split_labels[[i]]
    names(data)[i]<-paste0(names(data)[i],"<=",bestsplitvalue)
    for(j in 1:length(data[,bestfeature])){
      data[j,bestfeature]<-ifelse(data[j,bestfeature]<=bestsplitvalue,0,1)
    }
  }
  return(list(bestfeature,data))
}
#----------------------------------step 4递归产生决策树----------------------------------------------
#定义一个统计分类变量频次的函数，返回出现频次最多的值
major_class_fun<-function(classlist){
  return(names(sort(table(classlist),decreasing = T)[1]))
}
#递归产生决策树
#dataset:用于构建树的数据集,最开始就是data_full，然后随着划分的进行越来越小，第一次划分之前是17个瓜的数据在根节点，然后选择第一个bestFeat是纹理 
#--纹理的取值有清晰、模糊、稍糊三种，将瓜分成了清晰（9个），稍糊（5个），模糊（3个）,这个时候应该将划分的类别减少1以便于下次划分 
#label_set:用于划分的类别 
#data_full：全部的数据 
#label_full:全部的类别 

#递归终止条件有三个： 
#1、当前节点包含的样本全部属于同一类别 
#2、当前属性集为空，将当前节点作为叶子节点，类别设定为该结点所含样本类别最多的类
#3、当前节点所包含的样本集合为空。将当前结点标记为叶结点，类别设定为父结点所含样本最多的类别
create_tree_fun<-function(dataset){
  if(is.null(dim(dataset))==FALSE){
  classlist<-dataset[,ncol(dataset)]
  #终止条件1
  if(length(unique(classlist))==1){
    return(classlist[1])
  }
  bestfeature<-choose_feature_fun(dataset)[[1]]
  dataset<-choose_feature_fun(dataset)[[2]]
  label_set<-names(dataset)
  bestlabel<-label_set[bestfeature]
  label_set<-label_set[-bestfeature]
  mytree<-list()
  mytree[bestlabel]<-list()
  uniqueval<-unique(dataset[,bestfeature])
  length(mytree[[bestlabel]])<-length(uniqueval)
  for(i in 1:length(uniqueval)){
    var<-uniqueval[i]
    mytree[[bestlabel]][[var]]<-create_tree_fun(split_discrete_fun(dataset,bestfeature,var))
  }
  }else{
    #终止条件2
    classlist<-dataset
    return(major_class_fun(classlist))
  }
  
  return(mytree)
}



