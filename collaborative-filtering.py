# https://github.com/adrianbarwicki/collaborative-filtering-demo
# https://blog.csdn.net/pipisorry/article/details/51788955
# equation link
# http://divakalife.blogspot.com/2010/04/data-mining-collaborative-filtering.html
# user CF algorithm
# find rating-similar users for a user and predict the user's rating 
# mdofify source branch


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error




class recommendation_helpers:
    
    def __init__(self):
        pass
    # custom
    #import recommendation_helpers
    # calculating mean squere error
    # for pred
    # simple_user_prediction : [n_users, n_items]
    # simple_item_prediction : [n_items, n_users]
    def get_mse(pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)

    # cross validation
    # ? can we user from sklearn.model_selection import train_test_split to do
    # probaly not becasue it cannot be selected row-based
    # ? renaming ratings to user_item_matrix is better
    def train_test_split(ratings):

        # ratings  = user_item_matrix = np.zeros((n_users, n_items)), value store rating

        test = np.zeros(ratings.shape)
        train = ratings.copy()
        for user in xrange(ratings.shape[0]):

            # choose 10 rating for a single users
            # ratings[user, :].nonzero()[0] : get index of rating non zero for a special user
            # test_ratings. 10 of non-zeros ratings , these rating will be used as testing, copy to test array
            # in original array set it as zero
            test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
            train[user, test_ratings] = 0.
            test[user, test_ratings] = ratings[user, test_ratings]
        
        # Test and training are truly disjoint
        assert(np.all((train * test) == 0)) 
        return train, test

    # measure distance L2
    # we use cosine similarity algorithm)
    # ? renaming ratings to user_item_matrix is better
    def calc_similarity(ratings, kind='user', epsilon=1e-9):

        # ratings  = user_item_matrix = np.zeros((n_users, n_items)), value store rating

        # epsilon -> small number for handling dived-by-zero errors
        if kind == 'user':
            # sim : (n_users, n_users)
            sim = ratings.dot(ratings.T) + epsilon
        elif kind == 'item':
            sim = ratings.T.dot(ratings) + epsilon

        # diagonal is user A to user A s square, same user
        # extract it as array , sqrt it as a single column vector
        # [] add one dimention so becomes row array
        # norms: [1, n_users]
        norms = np.array([np.sqrt(np.diagonal(sim))])

        # = sum / (norms.dot(norms.T) )
        # by test
        #  norms / norms.T = [n_users, n_users ]
        # for each element in norms it divides n_users element in norms.T and create [n_users, 1]
        # ? i prefer to go wtih  sim /  norms.T.dot(norms) to form [n_users, n_users]

        return (sim / norms / norms.T)


    # ? renaming ratings to user_item_matrix_training is better
    # similarity : [n_users, n_users] if kind = 'user'
    # similarity : [n_items, n_items] if kind = 'item'
    def predict_simple(ratings, similarity, kind='user'):
        if kind == 'user':
            # user-baed CF, also predict user rating
            # axis = 1, row base sum
            # np.array([]) from a 2d array with [1, n_users] ,transpost to [n_users, 1]
            # similarity.dot(ratings) = [n_users, n_items]
            #  for each user, output its scores for each item  as  [n_users, n_items]
            return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif kind == 'item':
            # item-based CF, also predict user rating
            # axis = 1, row base sum
            # np.array([]) from a 2d array with [1, n_items]
            # ratings.dot(similarity) = [n_items, users]
            # for each user, output its scores for each item  as  [n_items, n_users]
            return ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])

    def predict_topk(ratings, similarity, kind='user', k=40):
        pred = np.zeros(ratings.shape)
        if kind == 'user':
            for i in xrange(ratings.shape[0]):
                top_k_users = [np.argsort(similarity[:,i])[:-k-1:-1]]
                for j in xrange(ratings.shape[1]):
                    pred[i, j] = similarity[i, :][top_k_users].dot(ratings[:, j][top_k_users]) 
                    pred[i, j] /= np.sum(np.abs(similarity[i, :][top_k_users]))
        if kind == 'item':
            for j in xrange(ratings.shape[1]):
                top_k_items = [np.argsort(similarity[:,j])[:-k-1:-1]]
                for i in xrange(ratings.shape[0]):
                    pred[i, j] = similarity[j, :][top_k_items].dot(ratings[i, :][top_k_items].T) 
                    pred[i, j] /= np.sum(np.abs(similarity[j, :][top_k_items]))        
    
        return pred

## Reading data
cols_names = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

# modify to my own data path
# using pandas to read ratings as a dataframe
#ratings = pd.read_csv('./u.data', sep='\t', names=cols_names, encoding='latin-1')
ratings = pd.read_csv('./collaborate-filter.data', sep='\t', names=cols_names, encoding='latin-1')

# direccly access column, deduplicateion, get shape
# http://pandas.pydata.org/pandas-docs/stable/tutorials.html
# using ratings['user_id'] should be find access
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.unique.html
# unique values
n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

print str(n_users) + ' users | ' + str(n_items) + ' items'

# Construct user-item matrix
user_item_matrix = np.zeros((n_users, n_items))

# Fill out the user-item matrix
# iterative data frame by tuples
# tuples start index is 1
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.itertuples.html
for row in ratings.itertuples():
    # 0-1 value index,
    # map from value to index 
    # userid,
    user_item_matrix[row[1]-1, row[2]-1] = row[3]

# Calculating sparcity (in %): How many entries of the user_item_matrix are defined?
# https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.nonzero.html
# [0] : output row index
sparsity = float(len(user_item_matrix.nonzero()[0]))
sparsity /= (user_item_matrix.shape[0] * user_item_matrix.shape[1])
sparsity *= 100

print 'Sparsity: {:4.2f}%'.format(sparsity)

# We will split our data into training and test sets by removing 10 ratings per user from the training set and placing them in the test set.
# ? can we user from sklearn.model_selection import train_test_split to do
# probaly not becasue it cannot be selected row-based
train, test = recommendation_helpers.train_test_split(user_item_matrix)

# measure distance L2 between users
# ? why needs to do this
user_similarity = recommendation_helpers.calc_similarity(train, kind='user')

numpy.savetxt('temp/user-similarity.txt', user_similarity, fmt='%f')

# measure distance L2 between items
item_similarity = recommendation_helpers.calc_similarity(train, kind='item')

numpy.savetxt('temp/item-similarity.txt', item_similarity, fmt='%f')

# we predict with an average over all users' and display it's prediction error

# similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
# simple_user_prediction : [n_users, n_items]
simple_user_prediction = recommendation_helpers.predict_simple(train, user_similarity, 'user')
# simple_item_prediction : [n_items, n_users] 
simple_item_prediction = recommendation_helpers.predict_simple(train, item_similarity, 'item')
print 'Simple User-based CF MSE: ' + str(recommendation_helpers.get_mse(simple_user_prediction, test))
print 'Simple Item-based CF MSE: ' + str(recommendation_helpers.get_mse(simple_item_prediction, test))

# we predict with an average over the k-most similar users' and display it's prediction error

mse_user = []
mse_item = []

for k1 in range(1, 80):
    if (k1 % 1 == 0):
        topk_user_prediction = recommendation_helpers.predict_topk(train, user_similarity, kind='user', k=k1)
        topk_item_prediction = recommendation_helpers.predict_topk(train, item_similarity, kind='item', k=k1)
        topk_item_error = recommendation_helpers.get_mse(topk_item_prediction, test)
        topk_user_error = recommendation_helpers.get_mse(topk_user_prediction, test)
        mse_user.append((k1, topk_user_error))
        mse_item.append((k1, topk_item_error))
        print 'Topk User-based CF MSE:' + str(topk_user_error)
        print 'Topk Item-based CF MSE:' + str(topk_item_error)

plt.plot([ x[0] for x in mse_item ], [ x[1] for x in mse_item ], 'bs', label='Top-k Item-Item')

plt.plot([ x[0] for x in mse_user ], [ x[1] for x in mse_user ], 'g^', label='Top-k User-User')

plt.xlabel('Top k')
plt.ylabel('MSE')
plt.legend()
plt.savefig('top-k-error')
