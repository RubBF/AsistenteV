# chatbot.py

import json
import numpy as np
import nltk
import pickle
import random
import tensorflow as tf
from nltk.stem import SnowballStemmer

# Inicializar stemmer en español
stemmer = SnowballStemmer('spanish')

# 1. Cargar datos (modelo, words, classes, e intenciones)
intents = json.loads(open('data/intents.json', encoding='utf-8').read())
words = pickle.load(open('data/words.pkl', 'rb'))
classes = pickle.load(open('data/classes.pkl', 'rb'))
model = tf.keras.models.load_model('data/chatbot_model.h5')

# 2. Funciones de preprocesamiento
def clean_text(sentence):
    # Tokenizar con nltk
    tokens = nltk.word_tokenize(sentence)
    # Stemmizar cada token y pasarlo a minúsculas
    tokens = [stemmer.stem(token.lower()) for token in tokens]
    return tokens

def bag_of_words(sentence, words):
    # Crear una bolsa de palabras (vector) del mismo tamaño que 'words'
    sentence_tokens = clean_text(sentence)
    bag = [0] * len(words)
    for st in sentence_tokens:
        for i, w in enumerate(words):
            if w == st:
                bag[i] = 1
    return np.array(bag)

# 3. Función para predecir la intención
def predict_intent(sentence):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]))[0]
    threshold = 0.25
    results = []
    for i, r in enumerate(res):
        if r > threshold:
            results.append((i, r))
    # Ordenar por probabilidad
    results.sort(key=lambda x: x[1], reverse=True)
    # Crear lista con nombres de intenciones y su probabilidad
    intents_list = []
    for r in results:
        intents_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return intents_list

# 4. Función para obtener la respuesta
def get_response(intents_list, intents_json):
    if not intents_list:
        return "Lo siento, no entendí tu pregunta."
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Lo siento, no entendí tu pregunta."

# 5. Si quisieras probarlo en consola
if __name__ == "__main__":
    print("Chatbot listo. Escribe 'salir' para terminar.")
    while True:
        user_input = input("Tú: ")
        if user_input.lower() in ["salir", "exit", "quit"]:
            print("Chatbot: ¡Hasta luego!")
            break
        intents_list = predict_intent(user_input)
        response = get_response(intents_list, intents)
        print("Chatbot:", response)
