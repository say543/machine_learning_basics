import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# custom
#import recommendation_helpers
# calculating mean squere error
def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

# cross validation
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], 
                                        size=10, 
                                        replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

# measure distance L2
def calc_similarity(ratings, kind='user', epsilon=1e-9):
    # epsilon -> small number for handling dived-by-zero errors
    if kind == 'user':
        sim = ratings.dot(ratings.T) + epsilon
    elif kind == 'item':
        sim = ratings.T.dot(ratings) + epsilon
    norms = np.array([np.sqrt(np.diagonal(sim))])
    return (sim / norms / norms.T)



def predict_simple(ratings, similarity, kind='user'):
    if kind == 'user':
        return similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif kind == 'item':
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
ratings = pd.read_csv('./u.data', sep='\t', names=cols_names, encoding='latin-1')


n_users = ratings.user_id.unique().shape[0]
n_items = ratings.movie_id.unique().shape[0]

print str(n_users) + ' users | ' + str(n_items) + ' items'

# Construct user-item matrix
user_item_matrix = numpy.zeros((n_users, n_items))

# Fill out the user-item matrix
for row in ratings.itertuples():
    user_item_matrix[row[1]-1, row[2]-1] = row[3]

# Calculating sparcity: How many entries of the user_item_matrix are defined?
sparsity = float(len(user_item_matrix.nonzero()[0]))
sparsity /= (user_item_matrix.shape[0] * user_item_matrix.shape[1])
sparsity *= 100

print 'Sparsity: {:4.2f}%'.format(sparsity)

# We will split our data into training and test sets by removing 10 ratings per user from the training set and placing them in the test set.
train, test = recommendation_helpers.train_test_split(user_item_matrix)

# measure distance L2 between users
user_similarity = recommendation_helpers.calc_similarity(train, kind='user')

numpy.savetxt('temp/user-similarity.txt', user_similarity, fmt='%f')

# measure distance L2 between items
item_similarity = recommendation_helpers.calc_similarity(train, kind='item')

numpy.savetxt('temp/item-similarity.txt', item_similarity, fmt='%f')

# we predict with an average over all users' and display it's prediction error

# similarity.dot(ratings) / np.array([np.abs(similarity).sum(axis=1)]).T
simple_user_prediction = recommendation_helpers.predict_simple(train, user_similarity, 'user')
simple_item_prediction = recommendation_helpers.predict_simple(train, item_similarity, 'item')
print 'Simple Item-based CF MSE: ' + str(recommendation_helpers.get_mse(simple_item_prediction, test))
print 'Simple User-based CF MSE: ' + str(recommendation_helpers.get_mse(simple_user_prediction, test))

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
