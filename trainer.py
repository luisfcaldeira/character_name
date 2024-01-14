from model import RNN, timeSince, randomTrainingExample, train, categoryTensor, inputTensor, unicodeToAscii, n_letters, all_letters
import time
import torch
import pandas as pd

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every ``plot_every`` ``iters``
pd_names = pd.read_csv('data/fictional_characters.csv', sep=';')

pd_names = pd_names.dropna()
categories = pd_names.gender.unique().tolist()
names = pd_names.character_name
all_categories = ['male', 'female']

category_lines = {}
category_lines['male'] = []
category_lines['female'] = []

n_categories = len(all_categories)

for index, row in pd_names.iterrows():    
    category_lines[row.gender].append(unicodeToAscii(row.character_name.strip()))




start = time.time()

rnn = RNN(n_letters, 128, n_letters, n_categories)

for iter in range(1, n_iters + 1):
    output, loss = train(*randomTrainingExample(all_categories, category_lines), rnn)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

torch.save(rnn, 'character_names.pth')
