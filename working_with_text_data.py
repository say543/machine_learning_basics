###
#https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# not yet finish
###



categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups

# this will download dattset from URI, remote
twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)

#output
twenty_train.target_names


len(twenty_train.data)

len(twenty_train.filenames)

#Text preprocessing, tokenizing and filtering of stopwords are all included in CountVectorizer, which builds a dictionary of features and transforms documents to feature vectors:
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape
