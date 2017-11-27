import numpy as np
import random

random.seed(0)

glove_dir = './data/glove.6B/'


def load_glove(size):
    path = glove_dir + 'glove.6B.' + str(size) + 'd.txt'
    wordvecs = {}
    with open(path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        for line in lines:
            tokens = line.split(' ')
            vec = np.array(tokens[1:], dtype=np.float32)
            wordvecs[tokens[0]] = vec

    return wordvecs


def fill_with_gloves(word_to_id, emb_size, wordvecs=None):
    if not wordvecs:
        wordvecs = load_glove(emb_size)

    n_words = len(word_to_id)
    res = np.zeros([n_words, emb_size], dtype=np.float32)
    n_not_found = 0
    for word, id in word_to_id.items():
        if isinstance(word, bytes):
            word = word.decode('utf-8')
        if word in wordvecs:
            res[id, :] = wordvecs[word]
        else:
            n_not_found += 1
            res[id, :] = np.random.normal(0.0, 0.1, emb_size)
    print('n words not found in glove word vectors: ' + str(n_not_found))

    return res

if __name__ == '__main__' :
    init_emb = fill_with_gloves(word_to_id, 100)
