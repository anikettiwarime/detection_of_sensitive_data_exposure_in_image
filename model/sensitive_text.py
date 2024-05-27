

import tensorflow as tf
import json
import os
import random

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Downloading the dataset
git_folder = "/content/drive/MyDrive"


dataset_folder = git_folder + "/text_dataset/"
sensitive_datafile = "SensitiveDataset.json"
nonsensitive_datafile = "NonSensitiveDataset.json"

# Necessary Variables
#hyperparameters and settings used throughout the code for tokenization, padding, and model training.
vocab_size = 3000
embedding_dim = 32
max_length = 70
truncation_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 20000

!pip install json
import json
dataList = []
sentences = []
labels = []
# Stopwords should be removed or excluded from the given text so that more
# focus can be given to those words which define the meaning of the text.
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

def loadDataset(filename):
  with open(dataset_folder + filename, 'r') as f:
      datastore = json.load(f)
  for item in datastore:
    sentence = item['data']
    label = item['is_sensitive']
    for word in stopwords: #Remove stop words in sentence
      token = " " + word + " "
      sentence = sentence.replace(token, " ")
    dataList.append([sentence, label])

# Loading both sensitive and non-sensitive dataset
loadDataset(sensitive_datafile)
loadDataset(nonsensitive_datafile)

# Shuffling the dataset randomly
random.shuffle(dataList)

# Dataset size: 31500 (approx)
print("Dataset Size: ", len(dataList))

# Dataset has both sentences and labels
for item in dataList:
  sentences.append(item[0])
  labels.append(item[1])

# Splitting up the total dataset
# Training size = 20000
# Validation size = 11500 (approx)
training_sentences = sentences[0:training_size]
validation_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
validation_labels = labels[training_size:]

print("Training Dataset Size: ", len(training_sentences))
print("Sample Training Data:", training_sentences[0])
print("Validation Dataset Size: ", len(validation_sentences))
print("Sample Validation Data:", validation_sentences[0])

import os
import json

# Define the directory where to save the word_index.json file
output_folder = "/content/"

# Ensure that the output folder exists, create it if it doesn't
os.makedirs(output_folder, exist_ok=True)

# Save the word index (Used for deploying in web application)


# Tokenizer takes the num_words (here vocab_size = 3000) maximum occuring unique words from the dataset.
# Anything out of these words will be treated as Out of Vocabulary(<oov>)
# It strips the punctutations and removes upper-case letters.

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Apply the tokenizer on training sentences and generate the word index
# Eg: word_index["the"] = 1; word_index["cat"] = 2; etc.
tokenizer.fit_on_texts(training_sentences)

# Save the word index (Used for deploying in web application)
word_index = tokenizer.word_index
print("Size of word index:", len(word_index))

with open(os.path.join(output_folder, "word_index.json"), "w") as outfile:
    json.dump(word_index, outfile)
    print("Saving the word index as JSON in:", os.path.join(output_folder, "word_index.json"))

# with open("word_index.json", "w") as outfile:
#     json.dump(word_index, outfile)
#     print("Saving the word index as JSON")

# Transforms each word in sentences to a sequence of integers based on the word_index
training_sequences = tokenizer.texts_to_sequences(training_sentences)
# To feed the text into neural network - sentences must be of the same length. Hence we'll be using padding.
# If the sentences are smaller than the maxlen, then we'll pad (Here, we are using post padding)
# If the sentences are larger than the maxlen, then we'll truncate (Here, we are using post truncation)

#training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)
training_padded = pad_sequences(training_sequences,padding=padding_type, truncating=truncation_type)

# Apply the same for validation data
validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

validation_padded = pad_sequences(validation_sequences, padding=padding_type, truncating=truncation_type)
validation_padded = pad_sequences(validation_sequences, padding=padding_type, truncating=truncation_type)
#validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)

# Convert to Numpy arrays, so as to get it to work with TensorFlow 2.x
import numpy as np
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
validation_padded = np.array(validation_padded)
validation_labels = np.array(validation_labels)

# import os

# # Get the current working directory
# current_dir = os.getcwd()

# # List the contents of the current directory
# print("Current directory:", current_dir)
# print("Contents of the directory:")
# for item in os.listdir(current_dir):
#     print("-", item)

# Callbacks to cancel training after reaching a desired accuracy
# This is done to avoid overfitting
DESIRED_ACCURACY = 0.999
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') > DESIRED_ACCURACY:
      print("Reached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# Sequential - defines a SEQUENCE of layers in the neural network.
#The specific architecture chosen (embedding + convolutional + pooling + dense layers) is commonly used for text classification tasks and
#has been shown to be effective in capturing relevant features from text data.
model = tf.keras.Sequential([
    # Embedding - Turns positive integers (indexes) into dense vectors of fixed size (here embedding_dim = 32).4
    #An embedding layer converts word indices into dense vectors, capturing semantic meaning of words in a lower-dimensional space.
   # tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
   tf.keras.layers.Embedding(vocab_size, embedding_dim),
    # 1D convolution layer - filter size = 128, convolution window = 5, activation fn = ReLU
    #A convolutional layer extracts features from the embedded representations.
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    # Global average pooling operation (Flattening)
    #A global average pooling layer reduces dimensionality and summarizes the extracted features.
    tf.keras.layers.GlobalAveragePooling1D(),
    # Regular densely-connected Neural Network layer with ReLU activation function.
    #Dense layers further process the features and make predictions.
    tf.keras.layers.Dense(24, activation='relu'),
    # Regular densely-connected Neural Network layer with sigmoid activation function.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model.compile - Configures the model for training.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
# Adam -  optimization algorithm used instead of the classical stochastic gradient descent procedure to update network weights.

# Display the summary of the model
model.summary()

num_epochs = 10

# model.fit - Train the model for a fixed number of epochs
history = model.fit(training_padded,
                    training_labels,
                    epochs=num_epochs,
                    validation_data=(
                        validation_padded,
                        validation_labels),
                    verbose=1)
                    #callbacks=[callbacks])

import matplotlib.pyplot as plt

# Plot the accuracy and loss functions
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

import seaborn
print('Confusion Matrix')
y_predicted = model.predict(validation_padded)
y_predicted_labels = y_predicted > 0.5

size = np.size(y_predicted_labels)
y_predicted_labels = y_predicted_labels.reshape(size, )

for i in range (1, 5):
  total = i * size // 4
  cm = tf.math.confusion_matrix(labels=validation_labels[0:total],predictions=y_predicted_labels[0:total])

  # Calculate accuracy
  cm_np = cm.numpy()
  conf_acc = (cm_np[0, 0] + cm_np[1, 1])/ np.sum(cm_np) * 100
  print("Accuracy for", str(total), "Test Data = ", conf_acc)

  # Plot the confusion matrix
  plt.figure(figsize = (10,7))
  seaborn.heatmap(cm, annot=True, fmt='d')
  plt.title("Confusion Matrix for " + str(total) + " Test Data")
  plt.xlabel('Predicted')
  plt.ylabel('Expected')

# Save and convert the model (Used for deploying in web application)
model.save('model/text_model.h5', save_format='h5', include_optimizer=False)
print("Saved the model successfully")

!apt-get -qq install virtualenv
!virtualenv -p python3 venv
!source venv/bin/activate
!pip install -q tensorflowjs
!tensorflowjs_converter --input_format=keras /content/model/text_model.h5


# output_folder = "/content/"
# Ensure that the output folder exists, create it if it doesn't
os.makedirs(output_folder, exist_ok=True)
print("Model converted to JSON successfully")
model_config = model.to_json()
model_config = json.loads(model_config)
model_config.pop('config')['layers'][0]['config'].pop('batch_input_shape')
with open(os.path.join(output_folder, "model_config.json"), "w") as outfile:
    json.dump(model_config, outfile)
    print("Saving the model config as JSON in:", os.path.join(output_folder, "model_config.json"))
# with open('model_config.json', 'w') as f:
#     f.write(model_config)

import tensorflow as tf
# Define a custom function to load the model without the input_length parameter
def load_model_without_input_length(filepath):
    # Load the model with custom objects (if any)
    model = tf.keras.models.load_model('model/text_model.h5', custom_objects=None, compile=True)
    return model

# Example usage
model = load_model_without_input_length('text_model.h5')

# Install necessary libraries
!pip install pytesseract
!sudo apt install tesseract-ocr
!sudo apt-get install tesseract-ocr-all

import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

# Function to perform OCR on the uploaded image
def perform_ocr(image, lang):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract
    extracted_text = pytesseract.image_to_string(gray_image, lang=lang)

    return extracted_text

# Function to handle file upload
def handle_file_upload(file_content):
    image = Image.open(io.BytesIO(file_content))
    image_array = np.array(image)
    extracted_text = perform_ocr(image_array, lang='eng+mar+hin+kan')  # English, Marathi, Hindi, Kannada
    return extracted_text

# User interface for uploading image
from google.colab import files
uploaded = files.upload()

# Perform OCR on the uploaded image and display the extracted text
if len(uploaded) > 0:
    for file_name, file_content in uploaded.items():
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            extracted_text = handle_file_upload(file_content)
            print("Extracted Text:")
            print(extracted_text)
        else:
            print("Invalid file format. Please upload an image with .jpg, .jpeg, or .png extension.")
else:
    print("No file uploaded.")

# Sample examples
#sentence = ["phone no-91 24843899", "आधार - आम आदमी का अधिकार 0000 2222 8945"]
sentence = [extracted_text]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)
predictions = model.predict(padded)
for i in range(len(predictions)):
  print(predictions[i][0])
  if predictions[i][0]>0.5:
    print("Sensitive - "+ sentence[i])
  else:
    print("Non-Sensitive - " + sentence[i] )



from google.colab import drive
drive.mount('/content/drive')