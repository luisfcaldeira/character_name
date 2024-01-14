from forecaster import samples
import torch
import torch.nn as nn
from flask import Flask


app = Flask(__name__)


@app.route("/")
def hello_world():
    rnn2 = torch.load('character_names.pth')
    result = samples('female', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return result[0]