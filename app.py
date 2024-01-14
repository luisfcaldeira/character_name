from forecaster import samples
from flask import Flask, jsonify, request
import random

app = Flask(__name__)


@app.route("/")
def hello_world():
    result = samples('female', random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')))
    return jsonify(result[0])