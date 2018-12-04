
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
    
    # Counting the occurance of each pair
    # (i, x) => (index, (tag, word))
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


# sequences => tag_sequence
def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
        counts the number of occurrences of the value in the sequences list. The sequences
        collection should be a 2-dimensional array.

        For example, if the tag NOUN appears 275558 times over all the input sequences,
        then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.

    output format:
                    out = {tag: occurance}
    """
    sequences = itertools.chain.from_iterable(sequences)
    
    # Counting the occurance
    cnt = Counter()
    for x in sequences:
        cnt[x] += 1
    #print("Counter: ", cnt, "\n")  
    
    # Convering the Counter to list - (sequence, occurance) like ('NOUN', 1)
    word_count = list(cnt.items()) 
    print("Word count: ", word_count, "\n")
    
    # Sum of the occurance (for normalization)
    sum_corpus = 0
    for i in word_count:
        sum_corpus += i[1]
    print("Total number of corpuses: ", sum_corpus, "\n")
    
    dictionary = dict()
    for index, i in enumerate(word_count):
        #word_count[index] = i / total_length
        dictionary[i[0]] = i[1] #/ sum_corpus
        
    
    #print("Word count: ", word_count, "\n")
    
    return dictionary

# from framework
basic_model = HiddenMarkovModel(name="base-hmm-tagger")

"""
TODO: create states with emission probability distributions P(word | tag) and add to the model
      (Hint: you may need to loop & create/add new states)
"""
# emission_probability calculation
# data.stream from whole set
tags = [tag for word, tag in data.stream()]
words = [word for word, tag in data.stream()]

# emission_counts : {tag: {word: occurance}}
emission_counts = pair_counts(tags, words)


# ? why unigram only using training_set instead of the whole set
# X => word_sequences
# Y => tag_sequences
tag_unigrams = unigram_counts(data.training_set.Y)
#print(tag_unigrams)
#print(tag_unigrams['NOUN'])

# sanity check for uni gram
assert set(tag_unigrams.keys()) == data.training_set.tagset, \
       "Uh oh. It looks like your tag counts doesn't include all the tags!"
assert min(tag_unigrams, key=tag_unigrams.get) == 'X', \
       "Hmmm...'X' is expected to be the least common class"
assert max(tag_unigrams, key=tag_unigrams.get) == 'NOUN', \
       "Hmmm...'NOUN' is expected to be the most common class"
HTML('<div class="alert alert-block alert-success">Your tag unigrams look good!</div>')



states = {}

for tag in data.tagset:
    emission_probabilities = dict()
    
    # noun, occurance => word, occurance
    for noun, occurance in emission_counts[tag].items(): # dict_items([('Mr.', 844), ('Podger', 21), ..])
        emission_probabilities[noun] = occurance / tag_unigrams[tag] # {'him': 2.5391666455069447e-05, 'he': 2.5391666455069447e-05, ...}
    
    # built in library ...
    tag_distribution = DiscreteDistribution(emission_probabilities) 
    #print(tag_distribution)
    
    # build in library: State class
    state = State(tag_distribution, name=tag)
    states[tag] = state
    basic_model.add_state(state)



"""
TODO: add edges between states for the observed transition frequencies P(tag_i | tag_i-1)
      (Hint: you may need to loop & add transitions
"""

def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.
    
    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093

    output format:
                    out = {tag: occurance}
    """
    list_of_start_tags = []
    for i in sequences:
        list_of_start_tags.append(i[0])
    #print("Bigram list: ", list_of_start_tags[-20:], "\n")
    
    # Counting the occurance
    cnt = Counter()
    for x in list_of_start_tags:
        cnt[x] += 1
    #print("Counter: ", cnt, "\n")  
    
    # Convering the Counter to list - (sequence, occurance) like ('NOUN', 1)
    word_count = list(cnt.items()) 
    #print("Word count: ", word_count, "\n")

    dictionary = dict()
    for index, i in enumerate(word_count):
        dictionary[i[0]] = i[1] 

    
    #print("Dictionary: ", dictionary, "\n")
    
    return dictionary



# TODO: Calculate the count of each tag starting a sequence
# X => word_sequences
# Y => tag_sequences
# from whole set
tag_starts = starting_counts(data.Y)
#print(tag_starts['NOUN'])

# sanity checking
assert len(tag_starts) == 12, "Uh oh. There should be 12 tags in your dictionary."
assert min(tag_starts, key=tag_starts.get) == 'X', "Hmmm...'X' is expected to be the least common starting bigram."
assert max(tag_starts, key=tag_starts.get) == 'DET', "Hmmm...'DET' is expected to be the most common starting bigram."
HTML('<div class="alert alert-block alert-success">Your starting tag counts look good!</div>')




# transition_probability calculation 

# Add start state
for tag in data.tagset:
    state = states[tag]
    #  START tag cnt is total tags cnt
    start_probability = tag_starts[tag] / sum(tag_starts.values())
    basic_model.add_transition(basic_model.start, state, start_probability)
    
    

def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.
    
    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    output format:
                    out = {tag: occurance}
    """
    # TODO: Finish this function!
    list_of_end_tags = []
    for i in sequences:
        list_of_end_tags.append(i[-1])
    #print("Bigram list: ", list_of_start_tags[-20:], "\n")
    
    # Counting the occurance
    cnt = Counter()
    for x in list_of_end_tags:
        cnt[x] += 1
    #print("Counter: ", cnt, "\n")  
    
    # Convering the Counter to list - (sequence, occurance) like ('NOUN', 1)
    word_count = list(cnt.items()) 
    #print("Word count: ", word_count, "\n")

    dictionary = dict()
    for index, i in enumerate(word_count):
        dictionary[i[0]] = i[1] 

    
    #print("Dictionary: ", dictionary, "\n")
    
    return dictionary



# TODO: Calculate the count of each tag ending a sequence
# X => word_sequences
# Y => tag_sequences
# from whole sets
tag_ends = ending_counts(data.Y)

assert len(tag_ends) == 12, "Uh oh. There should be 12 tags in your dictionary."
assert min(tag_ends, key=tag_ends.get) in ['X', 'CONJ'], "Hmmm...'X' or 'CONJ' should be the least common ending bigram."
assert max(tag_ends, key=tag_ends.get) == '.', "Hmmm...'.' is expected to be the most common ending bigram."
HTML('<div class="alert alert-block alert-success">Your ending tag counts look good!</div>')

# Add end state   
for tag in data.tagset:
    state = states[tag]

    # not always 1
    # ? but equation does not understand
    end_probability = tag_ends[tag] / sum(tag_ends.values())
    basic_model.add_transition(state, basic_model.end, end_probability)
    
    


def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.
    
    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582

    output format:
                    out = {{gram1, gram2}: occurance}
    """
    cnt = Counter()
    for seq in sequences:
        # if len >=2 , then should calculate bigram
        if len(seq) > 1:
            # index against index -1
            for index in range(1, len(seq)):
                bigram = (seq[index - 1], seq[index])
                if bigram not in cnt:
                    cnt[bigram] = 0
                cnt[bigram] += 1
    output = dict(cnt)
    
    return output


# X => word_sequences
# Y => tag_sequences
# from training set sets
# ? why only trainni
tag_bigrams = bigram_counts(data.training_set.Y)
#print(tag_bigrams)
#print(len(tag_bigrams))
#print(tag_bigrams[('NOUN', 'VERB')])

assert len(tag_bigrams) == 144, \
       "Uh oh. There should be 144 pairs of bigrams (12 tags x 12 tags)"
assert min(tag_bigrams, key=tag_bigrams.get) in [('X', 'NUM'), ('PRON', 'X')], \
       "Hmmm...The least common bigram should be one of ('X', 'NUM') or ('PRON', 'X')."
assert max(tag_bigrams, key=tag_bigrams.get) in [('DET', 'NOUN')], \
       "Hmmm...('DET', 'NOUN') is expected to be the most common bigram."
HTML('<div class="alert alert-block alert-success">Your tag bigrams look good!</div>')


# Add in between edges
for tag1 in data.tagset:
    state1 = states[tag1]
    sum_of_probabilities = 0
    for tag2 in data.tagset:
        state2 = states[tag2]
        # from a tuple
        bigram = (tag1,tag2)

        transition_probability = tag_bigrams[bigram] / tag_unigrams[tag1]
        sum_of_probabilities += transition_probability
        basic_model.add_transition(state1, state2, transition_probability)



#==============================================================
# finalize the model
#==============================================================


# NOTE: YOU SHOULD NOT NEED TO MODIFY ANYTHING BELOW THIS LINE
basic_model.bake()
print("Number of nodes or states: ", basic_model.node_count())
print("Number of edges: ", basic_model.edge_count())

assert all(tag in set(s.name for s in basic_model.states) for tag in data.training_set.tagset), \
       "Every state in your network should use the name of the associated tag, which must be one of the training set tags."
assert basic_model.edge_count() == 168, \
       ("Your network should have an edge from the start node to each state, one edge between every " +
        "pair of tags (states), and an edge from each state to the end node.")
HTML('<div class="alert alert-block alert-success">Your HMM network topology looks good!</div>')




#==============================================================
# evaluate train / test data set metrics
#==============================================================

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions


def accuracy(X, Y, model):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.
    
    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.
    
    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
        
        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions


# evaluation metrics for trainnig set and testing set
hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

assert hmm_training_acc > 0.97, "Uh oh. Your HMM accuracy on the training set doesn't look right."
assert hmm_training_acc > 0.955, "Uh oh. Your HMM accuracy on the training set doesn't look right."
HTML('<div class="alert alert-block alert-success">Your HMM tagger accuracy looks correct! Congratulations, you\'ve finished the project.</div>')


#==============================================================
# Example Decoding Sequences with the HMM Tagger
#==============================================================


for key in data.testing_set.keys[:3]:
    print("Sentence Key: {}\n".format(key))
    print("Predicted labels:\n-----------------")
    print(simplify_decoding(data.sentences[key].words, basic_model))
    print()
    print("Actual labels:\n--------------")
    print(data.sentences[key].tags)
    print("\n")
