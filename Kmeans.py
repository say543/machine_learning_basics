# not yet run through python environment

import numpy as np

import random
from sklearn.datasets import make_blobs
np.random.seed(123)

# install matplotlib
# this line for ipython mac only
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class KMeans():
    def __init__(self, n_clusters=4):
        self.k = n_clusters

    def fit(self, data):
        """
        Fits the k-means model to the given dataset
        """
        n_samples, _ = data.shape
        # initialize cluster centers
        # here select k sample datas as possible cluster centers
        self.centers = np.array(random.sample(list(data), self.k))
        # create a copy, not reference of a np.array
        self.initial_centers = np.copy(self.centers)

        # We will keep track of whether the assignment of data points
        # to the clusters has changed. If it stops changing, we are 
        # done fitting the model
        old_assigns = None
        n_iters = 0

        while True:
            # form a list
            new_assigns = [self.classify(datapoint) for datapoint in data]

            # list judges equality directly
            if new_assigns == old_assigns:
                print(f"Training finished after {n_iters} iterations!")
                return

            old_assigns = new_assigns
            n_iters += 1

            # recalculate centers
            for id_ in range(self.k):
                # list => array 
                # find index of arrays whose ids = id_
                points_idx = np.where(np.array(new_assigns) == id_)
                # get data points from a list of indexes to form a np array
                
                datapoints = data[points_idx]

                # debug 
                # numpy.ndarray
                #print(f'type: {type(datapoints)}')
                #print(f'type: {datapoints.shape}')

                self.centers[id_] = datapoints.mean(axis=0)

    def l2_distance(self, datapoint):
        dists = np.sqrt(np.sum((self.centers - datapoint)**2, axis=1))
        return dists

    def classify(self, datapoint):
        """
        Given a datapoint, compute the cluster closest to the
        datapoint. Return the cluster ID of that cluster.
        """
        dists = self.l2_distance(datapoint)
        return np.argmin(dists)

    def plot_clusters(self, data):
        plt.figure(figsize=(12,10))
        plt.title("Initial centers in black, final centers in red")
        plt.scatter(data[:, 0], data[:, 1], marker='.', c=y)
        plt.scatter(self.centers[:, 0], self.centers[:,1], c='r')
        plt.scatter(self.initial_centers[:, 0], self.initial_centers[:,1], c='k')
        plt.show()
        


# data set
X, y = make_blobs(centers=4, n_samples=1000)
print(f'Shape of dataset: {X.shape}')

fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c=y)
plt.title("Dataset with 4 clusters")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()
        
# initial and fit model
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)


# plot and find clsuter center
kmeans.plot_clusters(X)