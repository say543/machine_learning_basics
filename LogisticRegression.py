# not yet run through python environment

import numpy as np

import random
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
np.random.seed(123)

# install matplotlib
# this line for ipython mac only
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(self):
        pass

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def train(self, X, y_true, n_iters, learning_rate):
        """
        Trains the logistic regression model on given data X and targets y
        """
        # Step 0: Initialize the parameters
        n_samples, n_features = X.shape


        self.weights = np.zeros((n_features, 1))

        # ? why bias initial as zero
        # ? and it serves as a matrix
        self.bias = 0
        costs = []
        
        # termination condition is decide by n_iters, hyperparamter
        for i in range(n_iters):
            # Step 1 and 2: Compute a linear combination of the input features and weights, 
            # apply the sigmoid activation function
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)


            # debug 
            # numpy.ndarray
            # [num of samples X 1]
            print(f'type: {type(y_predict)}')
            print(f'type: {y_predict.shape}')

            
            # Step 3: Compute the cost over the whole training set.
            # cross entropy
            # cost is a single value
            # elementwise product+ np.sum becomes a single value
            cost = (- 1 / n_samples) * np.sum(y_true * np.log(y_predict) + (1 - y_true) * (np.log(1 - y_predict)))

            # Step 4: Compute the gradients
            # dw : [number of features] 
            dw = (1 / n_samples) * np.dot(X.T, (y_predict - y_true))

            # np.sum then single value
            db = (1 / n_samples) * np.sum(y_predict - y_true)

            # Step 5: Update the parameters
            self.weights = self.weights - learning_rate * dw
            self.bias = self.bias - learning_rate * db

            costs.append(cost)

            # print out cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")

        return self.weights, self.bias, costs

    def predict(self, X):
        """
        Predicts binary labels for a set of examples X.
        """
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

        # using 0.5 as thresold
        # y_predict_labels is a list
        y_predict_labels = [1 if elem > 0.5 else 0 for elem in y_predict]

        return np.array(y_predict_labels)[:, np.newaxis]
        


# data set
X, y_true = make_blobs(n_samples= 1000, centers=2)

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y_true)
plt.title("Dataset")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
        


# Reshape targets to get column vector with shape (n_samples, 1)
y_true = y_true[:, np.newaxis]
# Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_true)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')



# initializing and training the model
regressor = LogisticRegression()
# hyper parameter
# iters = eposh
# learnig_rate
w_trained, b_trained, costs = regressor.train(X_train, y_train, n_iters=600, learning_rate=0.009)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(600), costs)
plt.title("Development of cost over training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()


# testing the model
y_p_train = regressor.predict(X_train)
y_p_test = regressor.predict(X_test)

print(f"train accuracy: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
print(f"test accuracy: {100 - np.mean(np.abs(y_p_test - y_test))}%")