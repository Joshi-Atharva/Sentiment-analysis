from google.colab import drive
drive.mount('/content/drive')
# Mounting the drive to use the files and images
import os
import csv
os.chdir('/content/drive/MyDrive/Sentiment analysis/')
print(os.listdir())
# rom extract_csv import extract_csv, split
from extract_bin import extract_bin, split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

file_path = 'amazon_cells_labelled.txt'
data = extract_bin(file_path)
inputs = data.inputs
labels = data.labels
data = extract_bin("train_set_sentiment_final.txt")
input300 = data.inputs
labels300 = data.labels
inputs.extend(input300)
labels.extend(labels300)

split_object = split(inputs, labels, 1000)
train_inputs = split_object.train_set_inputs
train_labels = split_object.train_set_labels
test_inputs = split_object.test_set_inputs
test_labels = split_object.test_set_labels
print(train_inputs)
print(train_labels)

print(len(input300))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_inputs)
print(tokenizer.word_index)
tokenized_inputs = tokenizer.texts_to_sequences(train_inputs)
print(tokenized_inputs)


from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_inputs = pad_sequences(tokenized_inputs, padding='post')
train_labels = tf.convert_to_tensor(train_labels)
print(padded_inputs)


X = padded_inputs
Y = train_labels
print(X.shape)
print(Y.shape)
print(Y)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128))
model.add(LSTM(units=64, return_sequences=True))
model.add(LSTM(units=64))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# input_dim specify the vocab count, output_dim the dense vector dim, input_length the number of words in the sentence

model.fit(X, Y, epochs = 30)

want_to_load = int(input("want to load model? [1/0]"))
if(want_to_load == 1):
    model = tf.keras.models.load_model('model.keras')
    print("model loaded")
def predict(model, tokenizer, input):
  tokenized_inputs = tokenizer.texts_to_sequences(input)
  padded_inputs = pad_sequences(tokenized_inputs, padding='post')
  predictions = model.predict(padded_inputs)
  return predictions

accuracy = 0
for i in range(len(test_inputs)):
    print(i, ':', test_inputs[i])
    prediction = predict(model, tokenizer, [test_inputs[i]])
    print(prediction)

    if prediction > 0.5:
        predicted_label = 1
        print("Positive")
    else:
        predicted_label = 0
        print("Negative")
    if( predicted_label == test_labels[i]):
        accuracy += 1
    print(predict(model, tokenizer, [test_inputs[i]]))
accuracy = accuracy/len(test_inputs)
print(f'accuracy = {accuracy}')

text = "i though this was good but it is actually bad"
text_list = text.split()
for i in range(len(text_list)):
    if(text_list[i] not in tokenizer.word_index):
        print(text_list[i], ':not found')
for key, value in tokenizer.word_index.items():
    print(f'{key}:{value}')
prediction = predict(model, tokenizer, [text])
print(prediction)
if(prediction > 0.5):
    print("Positive")
else:
    print("Negative")

print(len(tokenizer.word_index))


tf.keras.models.save_model(model, 'model.keras')
os.listdir()

