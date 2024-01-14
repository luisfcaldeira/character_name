import torch
from model import categoryTensor, inputTensor, n_letters, all_letters

rnn2 = torch.load('character_names.pth')
all_categories = ['male', 'female']

n_categories = len(all_categories)


max_length = 20

# Sample from a category and starting letter
def sample(category, start_letter='A'):
    with torch.no_grad():  # no need to track history in sampling
        category_tensor = categoryTensor(category, n_categories=n_categories, all_categories=all_categories)
        input = inputTensor(start_letter)
        hidden = rnn2.initHidden()

        output_name = start_letter

        for i in range(max_length):
            output, hidden = rnn2(category_tensor, input[0], hidden)
            topv, topi = output.topk(1)
            topi = topi[0][0]
            if topi == n_letters - 1:
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)

        return output_name

# Get multiple samples from one category and multiple starting letters
def samples(category, start_letters='ABC'):
    result = []
    for start_letter in start_letters:
        result.append(sample(category, start_letter))

    return result

if __name__ == '__main__':
    print(samples('male', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))