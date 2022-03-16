import pandas as pd
import numpy as np
import io
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# from google.colab import files




import nltk





# TrainDF = pd.read_csv(io.BytesIO(uploaded['ex3_train_data.csv']))
# TestDF = pd.read_csv(io.BytesIO(uploaded['ex3_test_data.csv']))
# SubDF = pd.read_csv(io.BytesIO(uploaded['ex3_sampleSubmission.csv']))







from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence

from keras.layers import *
from keras.models import *
from keras.layers import Dense

data_train=pd.read_csv('ex3_train_data.csv',names=['sentence', 'label'])



sentences_train, sentences_test, y_train, y_test = train_test_split(
sentences, y, test_size=0.20, random_state=1000)

print(sentences_train.shape)
print(sentences_test.shape)
print(y_train.shape)
print(y_test.shape)



print(sentences_train.shape[0])






maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embedding_dim = 50


model = Sequential()

# model.add(Embedding())
# model.add(Embedding(1000,5,input_dim=input_dim))
# model.add(Embedding(1000,5,input_length=max_words))

model.add(Embedding(input_dim=vocab_size,
                           output_dim=embedding_dim,
                           input_length=maxlen))

# model.add(Conv1D(64,5,activation='relu', input_shape=(input_dim,1)))


model.add(Conv1D(64,5,activation='relu'))
model.add(MaxPool1D(2,2))
model.add(Conv1D(32,5,activation='relu'))
model.add(MaxPool1D(2,2))
model.add(Conv1D(16,5,activation='relu'))
model.add(MaxPool1D(2,2))
model.add(Conv1D(8,5,activation='relu'))
model.add(MaxPool1D(2,2))

# model.add(Conv1D(64,2,activation='relu'))
# model.add(MaxPool1D(2,2))
# model.add(Conv1D(32,2,activation='relu'))
# model.add(MaxPool1D(2,2))
# model.add(Conv1D(16,2,activation='relu'))
# model.add(MaxPool1D(2,2))
# model.add(Conv1D(8,2,activation='relu'))
# model.add(MaxPool1D(2,2))



# model.add(Flatten())
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# model.add(Flatten())


model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.compile(optimizer='adam', loss='mse', metrics=['acc'])

hist=model.fit(X_train, y_train, validation_split=0.2,epochs=10, batch_size=128)

X_train

score = model.evaluate(X_test, y_test, batch_size=128, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

sentences_f = TestDF['sentence'].values

fX_test = tokenizer.texts_to_sequences(sentences_f)
fX_test = pad_sequences(fX_test, padding='post', maxlen=maxlen)

fX_test

res=np.argmax(model.predict(fX_test, verbose=0), axis=1)
res.shape

res_l=[]
for i in range(len(res)):
  res_l.append(res[i][0])
  # if(res[i]==0):
  #   print(i)

SubDF.label=res_l
SubDF.to_csv('res3.csv',index=False)