
# link 
# https://medium.com/@jacky308082/machine-learning-%E4%B8%8B%E7%9A%84support-vector-machine%E5%AF%A6%E4%BD%9C-%E4%BD%BF%E7%94%A8python-3b1c0dc8639c
# https://github.com/llSourcell/Classifying_Data_Using_a_Support_Vector_Machine/blob/master/support_vector_machine_lesson.ipynb
# 

import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

class HyperPlane_helpers:
    
    def __init__(self):
        pass

    #hyperplane=x,w+b
    #v=x,w+b
    #正的value psv=1
    #負的value nsv=-1
    #剛好為0的value dec=0
    #在這邊因為是用x[:,0],x[:,1] 的方式來畫x,y軸
    #所以實際的方程式原本為此 v=w1*x[:,0]+w2*x[:,1]+b 進而求出w1,w2之間的關係
    def hyperplane_value(x,w1,w2,b,v):
        return (-w1*x-b+v) / w2

	# visualize points after training  and weight update
	# output hyper plane
	# def visualize(data_dict):
    def visualize(x,min_feature_value,max_feature_value,w):

        #[[ax.scatter(x[0],x[1],s=100,color=color[i])]
    	plt.scatter(x[:,0],x[:,1],marker='o',c=y)

    	#max_feature_value表示為在x1,x2的圖上最右或者最上的值 
    	#在這裡可以限制圖上的datapoint
    	datarange=(min_feature_value*0.9,max_feature_value*1.1)
    	hyp_x_min=datarange[0]
    	hyp_x_max=datarange[1]

    	#(w,x+b)=1 預測為1
    	#positive support vector hyperplane
    	psv1=HyperPlane_helpers.hyperplane_value(hyp_x_min,w[0],w[1],w[2],1)
    	psv2=HyperPlane_helpers.hyperplane_value(hyp_x_max,w[0],w[1],w[2],1)
    	plt.plot([hyp_x_min,hyp_x_max],[psv1,psv2],'k--')
    
    	#(w,x+b)=-1
    	#negative support vector hyperplane
    	nsv1=HyperPlane_helpers.hyperplane_value(hyp_x_min,w[0],w[1],w[2],-1)
    	nsv2=HyperPlane_helpers.hyperplane_value(hyp_x_max,w[0],w[1],w[2],-1)
    	plt.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2],'k--')
    
    	#(w,x+b)=0
    	#positive support vector hyperplane
    	db1=HyperPlane_helpers.hyperplane_value(hyp_x_min,w[0],w[1],w[2],0)
    	db2=HyperPlane_helpers.hyperplane_value(hyp_x_max,w[0],w[1],w[2],0)
    	plt.plot([hyp_x_min,hyp_x_max],[db1,db2],'k')
    
    	plt.show()


#引入資料集
# only 100 samples
x,y=make_blobs(n_samples=100,centers=2,random_state=0,cluster_std=0.8)

#建立plot 去看由第一個feature 和第二個feature組成圖
# x [num_of_samples, num_of_features(2)]
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='autumn')
# this is to drop picutre
plt.show()


# now label all {0, 1} before mapping
print(f'y_label: {y}')



###################################
# start training
###################################`


# mapping y {0,1 } to y{-1, 1}
for i in range(len(y)):
    if y[i]==0:
        y[i]=-1


#以正確的格式方入 for two features
# 100 becasue line 16 says only 100 samples
# ? using  better paramter in the future
train_f1 = x[:,0]
train_f2 = x[:,1]
# bias initial as 1 
train_b= np.ones((100,1))
train_f1 = train_f1.reshape(100,1)
train_f2 = train_f2.reshape(100,1)
train_total=np.c_[train_f1,train_f2,train_b]

#權重(要包括bias)
w = np.zeros(train_total.shape[1])

y_train = np.array(y)
y_train.reshape(100,1)

epochs=1
#此為learning rate
alpha=0.0005

while(epochs < 10000):

	# train_total [number of asmpke(100), number of featires(2) + bias(1)]
	# w : [number of featires(2) + bias(1)], 1]

    #hyperplane 預測y值
    y_new2 = np.dot(train_total,w)


	# ? 覺得ravel 好像沒有比較好 
	# 直接一維做就可以

    #求出>1, =1,  or <1
    prod = ((y_new2.ravel()) * (y_train.ravel()))

    print(epochs)


    # choose lambda 1/epochs
    # as epochs increases, lambda decreases
    # ? not sure why goes this way

    count = 0
    for val in prod:
      #檢測是否有>=1 如果有,就表示正確分類
        if(val >= 1):
            cost = 0
            w = w - alpha * (2 * 1/epochs * w)
      #沒有 表示為錯誤分類
        else:
            cost = 1 - val
            # w = w - alpha ( - yi**xi + 2 ** lambda * w ) 
            w = w + alpha * (train_total[count] * y_train[count] - 2 * 1/epochs * w)
        count += 1
    epochs += 1



###################################
# for hyper plane 的輸出
#現在我要建立分別取出最大和最小的值,可以供我等一下的bias建立, 可以更宏觀的建立最佳的bias
# ? 不太知道怎麼達成的
###################################`

postiveX=[]
negativeX=[]
for i,v in enumerate(y):

    # becasue y is mapped from {0,1} to {-1 ,1}
    if v==-1:
        negativeX.append(x[i])
    else:
        postiveX.append(x[i])    

    '''
    if v==0:
        negativeX.append(x[i])
    else:
        postiveX.append(x[i])
    '''

# for large margin, map y label {0,1} to {-1, 1}
#把為target為-1的data存入-1的values裡
#把為target為1的data存入1的values裡
data_dict = {-1:np.array(negativeX), 1:np.array(postiveX)}


#定義learning rate
#建立最小的min_feature_value
max_feature_value=float('-inf')
min_feature_value=float('inf')

#針對-1和1的target開始找
# ? 兩種features 同時找
for yi in data_dict:
    #找出最大值 如果在-1找到更大的 就會取代原本的最大值
    if np.amax(data_dict[yi])>max_feature_value:
        max_feature_value=np.amax(data_dict[yi])
    #找出最小值 方法如上
    if np.amin(data_dict[yi])<min_feature_value:
        min_feature_value=np.amin(data_dict[yi])

# output hyperplane
HyperPlane_helpers.visualize(x, min_feature_value, max_feature_value, w)

