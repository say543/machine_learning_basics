#https://github.com/say543/machine_learning_basics/blob/master/softmax_regression.ipynb


import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.datasets import make_blobs
np.random.seed(13)



class SoftmaxRegressor:

    def __init__(self):
        pass

    def train(self, X, y_true, n_classes, n_iters=10, learning_rate=0.1):
        """
        Trains a multinomial logistic regression model on given set of training data
        """
        self.n_samples, n_features = X.shape
        self.n_classes = n_classes
        
        # weight [number_clasees, number_features]
        self.weights = np.random.rand(self.n_classes, n_features)

        # bias [1, number_clasees]
        # column vector
        # ? bias 轉至感覺沒有必要
        # 這邊轉置是為了
        #  db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)
        # 這個link 可以直接加
        self.bias = np.zeros((1, self.n_classes))



        all_losses = []
        
        for i in range(n_iters):

            # for e^(w^t np.dot x) 轉成 e^(x np.dot w^t)
            # scores => x np.dot w^t
            # softmax =>e^(x np.dot w^t)

            # scores: numpy array of shape (n_samples, n_classes)
            # bias 也有用來計算 不用參數傳  直接在function 裡面用
            scores = self.compute_scores(X)


            # probs : (n_samples, n_classes) 
            probs = self.softmax(scores)


            # based on true lable
            # calculate one -hot encoding 
            # y_one_hot : (n_samples, n_classes) 
            y_one_hot = self.one_hot(y_true)


            # ? 不知道這個y_predict 是幹嘛的
            # 先comment 因為沒有用
            '''
            # based on training data
            # calculate yk for each sample 
            # ? 不知道這個y_predict 是幹嘛的
            # axis = 1
            # a00, a01, a02 as a sinlge one to perfrom np.argmax
            # so this is calculated persample
            # np.argmax(probs, axis=1) :  (n_samples, 1)
            # np.argmax 是輸出index
            # np.array 只要願算結果變成1維 都是變成row base
            # 所以[:, np.newaxis] 轉成column vector
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            '''


            # y_one_hot : (n_samples, n_classes) 
            # probs : (n_samples, n_classes) 
            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)

            # gradient descent update
            # ? divided by nuber of samples
            # this is added bt repo


            # weight [number_clasees, number_features]

            # bias [1, number_clasees]
            # X : (n_samples, number_features)
            # X.T : (number_features, n_samples)
            # y_one_hot : (n_samples, n_classes) 
            # probs : (n_samples, n_classes) 
            # np.dot(X.T, (probs - y_one_hot)) : [number_clasees, number_features]
            # for each class k it has weight k as column vector in weight

            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))


            # for bias
            # y_one_hot : (n_samples, n_classes) 
            # probs : (n_samples, n_classes) 
            # (probs - y_one_hot) = (n_samples, n_classes) 
            # 因為biase 是把X : (n_samples, number_features) => X : (n_samples, number_features+1)
            # 額外的feature 都是1 
            # so 不需要 X 的input 
            # so 對每一個class sum 所有 n_samples 得直
            # np.sum(probs - y_one_hot, axis=0) : 
            # np.array 因為只有一維  一定會變成row 
            # (1, n_classes)
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)



            # minimize cross entropy, 所以用minus 的
            self.weights = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            # output loss each 100 iteration
            if i % 100 == 0:
                print(f'Iteration number: {i}, loss: {np.round(loss, 4)}')

        return self.weights, self.bias, all_losses

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            numpy array of shape (n_samples, 1) with predicted classes
        """

        # for e^(w^t np.dot x) 轉成 e^(x np.dot w^t)
        # scores => x np.dot w^t
        # softmax =>e^(x np.dot w^t)

        # scores: numpy array of shape (n_samples, n_classes)
        # bias 也有用來計算 不用參數傳  直接在function 裡面用

        scores = self.compute_scores(X)
        probs = self.softmax(scores)

        # get the class with bigger probability
        # probs : (n_samples, n_classes) 
        # axis = 1
        # a00, a01, a02 as a sinlge one to perfrom np_sum
        # which is the same sample 
        # np.argmax returns index 
        # np.array 只要願算結果變成1維 都是變成row base
        # 利用 [:, np.newaxis] 轉持column array
        return np.argmax(probs, axis=1)[:, np.newaxis]

    def softmax(self, scores):
        """
        Tranforms matrix of predicted scores to matrix of probabilities
        
        Args:
            scores: numpy array of shape (n_samples, n_classes)
            with unnormalized scores
        Returns:
            softmax: numpy array of shape (n_samples, n_classes)
            with probabilities
        """


        # for each sample, for each class ,caclulate
        # np.exp(scores) : still (n_samples, n_classes)

        # axis = 1
        # a00, a01, a02 as a sinlge one to perfrom np_sum
        # which is the same sample 
        # sum_exp : still (n_samples, 1)

        # softmax = (n_samples, n_classes) / (n_samples, 1) = (n_samples, n_classes) 

        sum_exp = np.sum(np.exp(scores), axis=1, keepdims=True)
        softmax = np.exp(scores) / sum_exp
        
        return softmax

    def compute_scores(self, X):
        """
        Computes class-scores for samples in X
    
        #  e(x np.dot w^t) =   , 這邊function 沒有exponentail 的部分    
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            scores: numpy array of shape (n_samples, n_classes)
        """
        return np.dot(X, self.weights.T) + self.bias

    def cross_entropy(self, y_true, probs):

        # using negative
        # add log

        # ? divided by nuber of samples
        # this is added bt repo

        # 用到 * 而不是dp.dot
        # y_true :   (n_samples, n_classes)
        # np.log(probs) : (n_samples, n_classes)

        # 沒有給axis 就是全部element 加起來不管維度
        loss = - (1 / self.n_samples) * np.sum(y_true * np.log(probs))
        return loss

    def one_hot(self, y):
        """
        Tranforms vector y of labels to one-hot encoded matrix

        Args:
            y: numpy array of shape (n_samples, 1)
        Returns:
            one_hot: numpy array of shape (n_samples, n_classes)
        """

        one_hot = np.zeros((self.n_samples, self.n_classes))

        # using np.array to select elements of another np.array

        # first diemention index
        # np.arange(self.n_samples) : (1, n_samples)
        # row vectors (0,1,2.....n_samples-1) 

        # second dimention index
        # 


        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot



#prepare dataset
# default n_features =2, 2 dimention
X, y_true = make_blobs(centers=4, n_samples = 5000)


#debug
print(f'Shape X: {X.shape}')
print(f'Shape y_true: {y_true.shape}')

fig = plt.figure(figsize=(8,6))

# using label as color
plt.scatter(X[:,0], X[:,1], c=y_true)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()




# split train / test set

# an array 1 X n_samples becomes n_samples x 1
# reshape targets to get column vector with shape (n_samples, 1)
y_true = y_true[:, np.newaxis]

#debug
#print(f'Shape y_true after newaxis: {y_true.shape}')


# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_true)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')



# initialize and training models
regressor = SoftmaxRegressor()
w_trained, b_trained, loss = regressor.train(X_train, y_train, learning_rate=0.1, n_iters=800, n_classes=4)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(800), loss)
plt.title("Development of loss during training")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.show()

# testing samples
n_test_samples, _ = X_test.shape
# y_predict  : numpy array of shape (n_samples, 1) with predicted classes
y_predict = regressor.predict(X_test)
print(f"Classification accuracy on test set: {(np.sum(y_predict == y_test)/n_test_samples) * 100}%")