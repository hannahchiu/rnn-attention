import numpy as np
import pandas as pd
import jieba
import tensorflow as tf
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors


def get_sentiment_dataset(filename, batch_size, seq_len):
    tag_class = {'P': 0, 'N': 1}
    # Read in data
    data = pd.read_csv(filename)
    y = data['tag'].map(tag_class)
    # Tokenize
    data["text"] = [list(jieba.cut(text)) for text in data["text"]]
    # Map text to sequence of word-integers and pad
    words = [word for text in data['text'] for word in text]
    words = sorted(set(words), key=words.index)
    word2idx = {c: i + 1 for i, c in enumerate(words)}
    X = []
    for text in data['text']:
        word_seq = []
        text_len = len(text)
        for i in range(seq_len):
            if i < text_len:
                word_seq.append(word2idx[text[i]])
            else:  # padding
                word_seq.append(0)
        X.append(word_seq)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create datasets
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.shuffle(10000)
    train_data = train_data.batch(batch_size)

    test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    train_data = train_data.shuffle(10000)
    test_data = test_data.batch(batch_size)

    return train_data, test_data, word2idx, len(X_train), len(X_test)


def load_pretrained_embed(embed_file, embed_size, word2idx):
    model = KeyedVectors.load_word2vec_format(embed_file, binary=True)
    embedding_matrix = np.zeros((len(word2idx) + 1, embed_size))
    num = 0
    for word, i in word2idx.items():
        if word in model:
            embedding_vector = model[word]
            embedding_matrix[i] = embedding_vector
        else:
            num = num + 1
            # print(word)
    print('覆蓋率:%.4f' % (1 - (num / len(word2idx))))
    return embedding_matrix
