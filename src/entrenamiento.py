# entrenamiento.py

import os
import json
import nltk
import pickle
import random
import numpy as np
import tensorflow as tf
from nltk.stem import SnowballStemmer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import nltk
nltk.download('punkt')


# Descarga (una vez) el tokenizador de NLTK para asegurarte de tener 'punkt'
# Descomenta si es la primera vez que ejecutas en tu entorno:
# nltk.download('punkt')

# 1. Cargar intenciones
with open('data/intents.json', 'r', encoding='utf-8') as file:
    intents = json.load(file)

stemmer = SnowballStemmer('spanish')
words = []
classes = []
documents = []

# Lista de caracteres que podemos ignorar
ignore_chars = ['?', '!', '.', ',', '¿', '¡']

# 2. Procesar cada intención y cada patrón
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenizar
        tokens = nltk.word_tokenize(pattern)
        # Limpiar y stemmatizar cada token
        tokens = [stemmer.stem(t.lower()) for t in tokens if t not in ignore_chars]
        # Agregar a la lista de palabras
        words.extend(tokens)
        # Guardar (tokens, tag) para el entrenamiento
        documents.append((tokens, intent['tag']))
        
        # Si la etiqueta no está en la lista de clases, agregarla
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Eliminar duplicados y ordenar
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# 3. Crear datos de entrenamiento
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    # Lista de palabras tokenizadas para el patrón
    token_words = doc[0]
    # Crear bag of words
    for w in words:
        bag.append(1) if w in token_words else bag.append(0)

    # Etiqueta
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # Agregar a training
    training.append([bag, output_row])

# Mezclar datos y convertir a array
random.shuffle(training)
training = np.array(training, dtype=object)

# Separar X e y
train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# 4. Construir la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compilar modelo
sgd = SGD(learning_rate=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# 5. Entrenar modelo
model.fit(train_x, train_y, epochs=100, batch_size=8, verbose=1)

# 6. Guardar modelo y datos
if not os.path.exists('data'):
    os.makedirs('data')

model.save('data/chatbot_model.h5')
pickle.dump(words, open('data/words.pkl', 'wb'))
pickle.dump(classes, open('data/classes.pkl', 'wb'))

print("Entrenamiento completado. Archivos guardados en la carpeta 'data'.")
