# app.py
from flask import Flask, request, jsonify, render_template
import os
from chatbot import predict_intent, get_response, intents

app = Flask(__name__)

@app.route("/")
def home():
    # Renderiza la página index.html desde la carpeta templates
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    # Obtenemos el texto que manda el usuario
    data = request.get_json()
    user_input = data.get("message", "")

    # Obtenemos la intención y la respuesta
    intents_list = predict_intent(user_input)
    response = get_response(intents_list, intents)
    
    # Devolvemos la respuesta en formato JSON
    return jsonify({"reply": response})

if __name__ == "__main__":
    # Ejecutamos la app en modo debug, ajusta el host/puerto si deseas exponerlo
    app.run(debug=True, port=5000)
