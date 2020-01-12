import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import re
import pickle
import tensorflow as tf
import os.path
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from collections import Counter

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping
from keras.layers import GlobalMaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D

from numpy import array
from numpy import asarray
from numpy import zeros

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)

def read_data(path, verbose):
    reviews = pd.read_csv(path)
    if verbose:
        print("Null values: ", reviews.isnull().values.any())
        print("Shape: ", reviews.shape)
        print("\nEXAMPLE DATA")
        print(reviews.head())
    return reviews


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')


def remove_tags(text):
    return TAG_RE.sub('', text)


def add_reviews_to_list():
    X = []
    sentences = list(reviews['review'])
    for s in sentences:
        X.append(preprocess_text(s))
    return X


def split_test_train(features, labels, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, 
    random_state=random_state)
    return X_train, X_test, y_train, y_test


def pickle_parameters(to_pickle, name):
    pickle_parameter = open(name + ".pickle","wb")
    pickle.dump(to_pickle, pickle_parameter)
    pickle_parameter.close()


def load_parameters(unpickle):
    pickle_in = open(unpickle + ".pickle","rb")
    parameter = pickle.load(pickle_in)
    return parameter


def graph_model_results(model, path, name):
    graphs = plt.figure(figsize=(13, 5))
    plt.subplot(1, 2, 1)
    plt.plot(model.history['acc'])
    plt.plot(model.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    plt.savefig(path + name + ".png")


def prepare_predictions(review):
    review = tokenizer.texts_to_sequences(review)

    flat_list = []
    for sublist in review:
        for item in sublist:
            flat_list.append(item)

    flat_list = [flat_list]

    review = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    return review

def simple_nn():
    model_name = 'simple-nn'

    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, 
    trainable=False)

    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    if os.path.isfile(model_name + '.pickle'):
        print('Model was saved. Loading saved model...')
        history = load_parameters("simple-nn")
    else:
        history = model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_split=0.2, 
        callbacks=[es])
        pickle_parameters(history, model_name)


    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    graph_model_results(history, "simple-nn/", model_name)


def convolutional_nn():
    model_name = 'convolutional_nn'

    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , 
    trainable=False)

    model.add(embedding_layer)
    model.add(Conv1D(128, 5, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    if os.path.isfile(model_name + '.pickle'):
        print('Model was saved. Loading saved model...')
        history = load_parameters("convolutional-nn")
    else:
        history = model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_split=0.2, 
        callbacks=[es])
        pickle_parameters(history, model_name)
    
    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    graph_model_results(history, "convolutional-nn/", model_name)


def recurrent_nn():
    model_name = 'recurrent-nn'
    
    model = Sequential()

    embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , 
    trainable=False)
    model.add(embedding_layer)
    model.add(LSTM(128))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())

    if os.path.isfile(model_name + '.pickle'):
        print('Model was saved. Loading saved model...')
        history = load_parameters(model_name)
    else:
        history = model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_split=0.2, 
        callbacks=[es])
        pickle_parameters(model, model_name)

    score = model.evaluate(X_test, y_test, verbose=1)
    print("Test Score:", score[0])
    print("Test Accuracy:", score[1])

    
    # graph_model_results(history, "recurrent-nn/", model_name)


reviews = read_data("imdb-dataset-of-50k-movie-reviews\IMDB Dataset.csv", verbose=False)

# X = add_reviews_to_list()
# y = np.array(list(reviews['sentiment'].map({'positive' : 1, 'negative' : 0})))

X = load_parameters("features")
y = load_parameters("labels")

# pickle_parameters(X, "features")
# pickle_parameters(y, "labels")

X_train, X_test, y_train, y_test = split_test_train(X, y, 0.02, 42)

print("No. of reviews: ", len(X))

# Assigning common words low numbers. e.g. the -> 0
combined_reviews = ' '.join(X)
words = combined_reviews.split()
count_words = Counter(words)

total_words = len(words)
sorted_words = count_words.most_common(total_words)

# DEBUGGING -- Don't Delete
# print("Total words: ", total_words)
# for i in range(10):
#     print(str(i) + ":", sorted_words[i])
# END

vocab_to_int = {w:i + 1 for i, (w,c) in enumerate(sorted_words)}

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(vocab_to_int)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open("glove\glove.6B.100d.txt", encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#y = np.array(list(map(lambda x: 1 if x=="positive" else 0, reviews['sentiment'])))
#reviews['sentiment'] = reviews['sentiment'].map({'positive' : 1, 'negative' : 0})

# NEURAL NETWORKS
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# SIMPLE NN
# simple_nn()

# CONVOLUTIONAL NN
# convolutional_nn()

# RECURRENT NN
# recurrent_nn()

# recurrent_nn(review1)

# print("Review 2: ", model.predict(prepare_predictions("Star Wars Episode IX is at times a decent thrill ride and it provides us with some satisfying moments but it fails to tell a compelling story It feels empty It rushes through a corporate checklist of must-have moments at a chaotic pace It manages to thrill with epic space battles but it lacks a soul There no creative vision here no story to tell The pressure that comes with making a Star Wars movie seems to have shackled the creators preventing them from doing anything interesting with the film It a shame The Star Wars universe is one of the greatest fictional universes ever created It should provide fertile ground for many new and interesting stories If I were Disney I would take a serious look at the creative process for these films They could probably learn a thing or two from Marvel Studios")))
# print("Review 3: ", model.predict(prepare_predictions("The movie has some good humorous moments but generally somewhat lengthy Also the story could have been a little more deepened")))
# print("Review 4: ", model.predict(prepare_predictions("Dont listen to the critics saying this movie is boring This movie is one of the most tense and exciting movies I seen in years Amazing cinematography and overall amazing experience of a movie")))