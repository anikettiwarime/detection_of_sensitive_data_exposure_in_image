from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from tensorflow.keras.models import load_model
import pytesseract
from PIL import Image
import io

# Load the model and tokenizer
model = load_model('model/text_model.h5')

# Load the tokenizer
with open('model/word_index.json', 'r') as f:
    word_index = json.load(f)
tokenizer = Tokenizer(num_words=3000, oov_token="<OOV>")
tokenizer.word_index = word_index

# Parameters for text preprocessing
max_length = 70
padding_type='post'
truncation_type='post'

# Function to preprocess text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=truncation_type)
    return padded

# Function to perform OCR on an uploaded image
def perform_ocr(image):
    gray_image = image.convert('L')  # Convert image to grayscale
    extracted_text = pytesseract.image_to_string(gray_image, lang='eng+mar+hin+kan')
    return extracted_text

@csrf_exempt
def home_view(request):
    prediction = None
    if request.method == 'POST':
        if 'file' in request.FILES and request.FILES['file']:
            # Get the image input and extract text
            file = request.FILES['file']
            image = Image.open(file)
            text = perform_ocr(image)

            # Preprocess the text and make prediction
            padded_text = preprocess_text(text)
            prediction = model.predict(padded_text)
            sensitivity = 'Sensitive' if prediction[0][0] > 0.5 else 'Non-Sensitive'

            prediction = {'text': text, 'sensitivity': sensitivity}
        else:
            return render(request, 'index.html', {'error': 'No valid image provided'})

    return render(request, 'index.html', {'prediction': prediction})
