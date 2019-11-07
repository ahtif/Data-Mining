import os
import pandas as pd
from pathlib import Path

"""
Step a
"""

#NOTE: Assuming same directory structure as the one from the stanford link
TRAIN_DIR = "./aclImdb/train/"
TEST_DIR = "./aclImdb/test/"

train = pd.DataFrame(data=None, columns=["text", "class"])
test = pd.DataFrame(data=None, columns=["text", "class"])

for file in os.listdir(TRAIN_DIR+"pos"):
    text=Path(TRAIN_DIR+"pos/"+file).read_text()
    train = train.append({"text":text, "class": "positive"}, ignore_index=True)

for file in os.listdir(TRAIN_DIR+"neg"):
    text=Path(TRAIN_DIR+"neg/"+file).read_text()
    train = train.append({"text":text, "class": "negative"}, ignore_index=True)

for file in os.listdir(TEST_DIR+"pos"):
    text=Path(TEST_DIR+"pos/"+file).read_text()
    test = test.append({"text":text, "class": "positive"}, ignore_index=True)
    
for file in os.listdir(TEST_DIR+"neg"):
    text=Path(TEST_DIR+"neg/"+file).read_text()
    test = test.append({"text":text, "class": "negative"}, ignore_index=True)

"""
Step b
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

VOCAB_SIZE = 5000
tokeniser = Tokenizer(num_words=VOCAB_SIZE)
tokeniser.fit_on_texts(train["text"])

#print(train)
#print(test)

train_data = train.copy()
test_data = test.copy()


train_data = train_data.head(500).append(train_data.tail(500))
train_data["text"] = tokeniser.texts_to_sequences(train_data["text"])
x_train = pad_sequences(train_data["text"], 100, dtype="int32", padding="post", truncating="post", value=0)

train_data["class"] = train_data["class"].apply(lambda x: 1 if x == "positive" else 0)
y_train = train_data["class"].values
#train_data.reset_index(drop=True)

test_data["text"] = tokeniser.texts_to_sequences(test_data["text"])
x_test = pad_sequences(test_data["text"], maxlen=100, dtype="int32", padding="post", truncating="post", value=0)
test_data["class"] = test_data["class"].apply(lambda x: 1 if x == "positive" else 0)
y_test = test_data["class"].values


"""
Step c
"""
import numpy as np
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

embedding_index = loadGloveModel("glove.6B.100d.txt")
embedding_matrix = np.zeros((VOCAB_SIZE+1, 100))
print(len(embeddings_index))
print(len(embedding_matrix))
print(embedding_matrix.shape)
tokeniser.word_index = dict(list(tokeniser.word_index.items())[:VOCAB_SIZE])
for word, i in tokeniser.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

"""
Step d
"""

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(VOCAB_SIZE+1, 100, weights=[embedding_matrix], input_length=100, trainable=False))
model.add(Flatten())
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
# write your code here
print(model.summary())
hist = model.fit(x_train, y_train, epochs=30, verbose=1)

"""
Step e
"""

import matplotlib.pyplot as plt

plt.figure(figsize=[8,6])
plt.plot(hist.history['loss'],'r',linewidth=3.0)
#plt.plot(hist.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss'],fontsize=12)
plt.xlabel('Epochs ',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.title('Loss Curves',fontsize=16)
plt.show()

plt.figure(figsize=[8,6])
plt.plot(hist.history['acc'],'r',linewidth=3.0)
#plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy'],fontsize=12)
plt.xlabel('Epochs ',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.title('Accuracy Curves',fontsize=16)
plt.show()


loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print('Accuracy on test set: %f' % (accuracy*100))


"""
Step f
"""

from keras.layers import LSTM

model2 = Sequential()
model2.add(Embedding(VOCAB_SIZE+1, 100, weights=[embedding_matrix], input_length=100, trainable=False))
model2.add(LSTM(100))
#model2.add(Flatten())
model2.add(Dense(1, activation="sigmoid"))

model2.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])
# write your code here
print(model2.summary())
hist2 = model2.fit(x_train, y_train, epochs=30, verbose=1)

plt.figure(figsize=[8,6])
plt.plot(hist2.history['loss'],'r',linewidth=3.0)
#plt.plot(hist.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss'],fontsize=12)
plt.xlabel('Epochs ',fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.title('Loss Curves',fontsize=16)
plt.show()

plt.figure(figsize=[8,6])
plt.plot(hist2.history['acc'],'r',linewidth=3.0)
#plt.plot(hist.history['val_acc'],'b',linewidth=3.0)
plt.legend(['Training Accuracy'],fontsize=12)
plt.xlabel('Epochs ',fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.title('Accuracy Curves',fontsize=16)
plt.show()


loss, accuracy = model2.evaluate(x_test, y_test, verbose=1)
print('Accuracy on test set: %f' % (accuracy*100))
