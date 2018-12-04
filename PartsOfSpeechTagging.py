
#https://github.com/soheillll/Parts-of-Speech-Tagging/blob/master/HMM%20Tagger.ipynb


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


# ? comment them at first since i do not think they are necessary 
# Jupyter "magic methods" -- only need to be run once per kernel restart
#%load_ext autoreload
#%aimport helpers, tests
#%autoreload 1

from itertools import chain
from collections import Counter, defaultdict
from helpers import show_model, Dataset

# ? replace with scikit learn in the future
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from IPython.core.display import HTML


import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import networkx as nx
import random

from io import BytesIO

# for sentence iteration
from itertools import chain
# for dataset parsing
from collections import namedtuple, OrderedDict



def read_data(filename):
  
    # move inside since it is only used by reading data
    Sentence = namedtuple("Sentence", "words tags")
  
  
    """Read tagged sentence data"""
    with open(filename, 'r') as f:
        sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
    return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
for l in s[1:]]))) for s in sentence_lines if s[0]))

def read_tags(filename):
    """Read a list of word tag classes"""
    with open(filename, 'r') as f:
        tags = f.read().split("\n")
return frozenset(tags)


class Subset(namedtuple("BaseSet", "sentences keys vocab X tagset Y N stream")):
    def __new__(cls, sentences, keys):
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        wordset = frozenset(chain(*word_sequences))
        tagset = frozenset(chain(*tag_sequences))
        N = sum(1 for _ in chain(*(sentences[k].words for k in keys)))
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        return super().__new__(cls, {k: sentences[k] for k in keys}, keys, wordset, word_sequences,
                               tagset, tag_sequences, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
return iter(self.sentences.items())


# dataset class to parsing
class Dataset(namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")):
    def __new__(cls, tagfile, datafile, train_test_split=0.8, seed=112890):
      
        # read tags
        tagset = read_tags(tagfile)
        
        # read setences
        sentences = read_data(datafile)
        
        # tuple : store all unique sentence identifier
        # ? might be reaname to sentenceKey
        keys = tuple(sentences.keys())
        
        # deduplicate to get unique words based on words in sentence
        # [] : a list
        wordset = frozenset(chain(*[s.words for s in sentences.values()]))
        
        
        # for each key, generating a word sequence as an element of tuple
        # for each key, generating a tag sequence as an element of tuple
        word_sequences = tuple([sentences[k].words for k in keys])
        tag_sequences = tuple([sentences[k].tags for k in keys])
        
        # ? total of words in dataset
        # duplication will count more than 1
        N = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        
        # split data into train/test sets
        # ? replaced by scikit learn
        _keys = list(keys)
        
        # initialize random seed
        if seed is not None: random.seed(seed)
          
        random.shuffle(_keys)
        
        
        # split is the index to seperate trainnig set / testing set
        split = int(train_test_split * len(_keys))
        
        # ? need to further understand what datastrucutre subset outputs
        training_data = Subset(sentences, _keys[:split])
        testing_data = Subset(sentences, _keys[split:])
        
        # for each element in stream tuple, it has word_sequence / tag_sequence as a pair
        # ? might to rename to wordseqTagseqPairTuple
        stream = tuple(zip(chain(*word_sequences), chain(*tag_sequences)))
        
        # https://www.jianshu.com/p/e938a06a85f4
        # namedtuple("_Dataset", "sentences keys vocab X tagset Y training_set testing_set N stream")
        # using this key order to access data
        # "sentences keys vocab X tagset Y training_set testing_set N stream" will be broken in different keys
        # _Dataset => cls
        # sentences => dict(sentences)
        # keys => keys
        # vocab => wordset
        # X => word_sequences
        # tagset => tagset
        # Y => tag_sequences
        # training_set => training_data
        # testing_set => testing_data
        # N => N
        # stream => stream.__iter__
        
        
        return super().__new__(cls, dict(sentences), keys, wordset, word_sequences, tagset,
                               tag_sequences, training_data, testing_data, N, stream.__iter__)

    def __len__(self):
        return len(self.sentences)

    def __iter__(self):
        return iter(self.sentences.items())



# read dataset
# tag list , dataset: traing data
# ? this can be replaced by scikit learn library
data = Dataset("tags-universal.txt", "brown-universal.txt", train_test_split=0.8)

# output statistics
print("There are {} sentences in the corpus.".format(len(data)))
print("There are {} sentences in the training set.".format(len(data.training_set)))
print("There are {} sentences in the testing set.".format(len(data.testing_set)))

# spliting size checking 
assert len(data) == len(data.training_set) + len(data.testing_set), \
       "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"

# output from Data class

# example output
# each ket to a unique sentence identifier
# each sentence is an object with two objects
# words: a tuple of the words in the sentence
# tags: a tupe of the tags corresponding.
key = 'b100-38532'
print("Sentence: {}".format(key))
print("words:\n\t{!s}".format(data.sentences[key].words))
print("tags:\n\t{!s}".format(data.sentences[key].tags))


# for dataset
# vocab: unique words
# tagset: unique sets
# all
print("There are a total of {} samples of {} unique words in the corpus."
      .format(data.N, len(data.vocab)))
# training set 
print("There are {} samples of {} unique words in the training set."
      .format(data.training_set.N, len(data.training_set.vocab)))
# testing set
print("There are {} samples of {} unique words in the testing set."
      .format(data.testing_set.N, len(data.testing_set.vocab)))

# split might create imbalance or missing vocab in testing
print("There are {} words in the test set that are missing in the training set."
      .format(len(data.testing_set.vocab - data.training_set.vocab)))

# sanity checking
assert data.N == data.training_set.N + data.testing_set.N, \
       "The number of training + test samples should sum to the total number of samples"



# output sentence for sanity checking 
        # X => word_sequences
        # Y => tag_sequences
# accessing words with Dataset.X and tags with Dataset.Y 
for i in range(2):    
    print("Sentence {}:".format(i + 1), data.X[i])
    print()
    print("Labels {}:".format(i + 1), data.Y[i])
    print()

# iterator pair 
# use Dataset.stream() (word, tag) samples for the entire corpus
# output five examples
print("\nStream (word, tag) pairs:\n")
for i, pair in enumerate(data.stream()):
    print("\t", pair)
    if i > 5: break



############################      
#build basic HMM tagger
############################




# sequenceA => tags
# seqenceB => words
# ? might be renaming to a better name
def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.
    
    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    
    output format:
                    out = {tag: {word: occurance}}
    """
    # Initialization
    dic = defaultdict(list)
    cnt = Counter()
    
    # Counting the occurance
    for i, x in enumerate(zip(sequences_A, sequences_B)):
        cnt[x] += 1
    
    # Convering the Counter to list - ((tag, word), occurance)
    tag_word_count = list(cnt.items()) 
    
    # Making a the main structure - {tag: [{word: occurance}]}
    for i in tag_word_count:
        dic[i[0][0]].append({i[0][1]: i[1]})
    
    out = dict(dic.items())
    
    # Convering the list of dictionaries to one dictionary - {tag: {word: occurance}}
    for tg in out:
        new_list = {}
        a = dic[tg] # a = dic['NOUN']
        
        # Convering list of dictionaries to one dictionary
        for index, i in enumerate(a):     
            new_list["".join(i.keys())] = list(i.values())[0]
            
        # Assiging to the new_list to 'out'
        out[tg] = new_list
        
    return out

# from framework
basic_model = HiddenMarkovModel(name="base-hmm-tagger")

"""
TODO: create states with emission probability distributions P(word | tag) and add to the model
      (Hint: you may need to loop & create/add new states)
"""


tags = [tag for word, tag in data.stream()]
words = [word for word, tag in data.stream()]
emission_counts = pair_counts(tags, words)







