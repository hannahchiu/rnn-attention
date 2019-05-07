from train import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = RnnAttention()
model.build()

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, model.model_path)
    sess.run(model.test_init)
    alpha, text = sess.run([model.alpha, model.text])

# get index and weight of first testing data
weight = alpha[0]
index_word = text[0]

# map sequence of index to word
word2idx = model.word2idx
word2idx["X"] = 0
words = []
for i in index_word:
    word = list(word2idx.keys())[list(word2idx.values()).index(i)]
    words.append(word)

# heatmap
df = pd.DataFrame({"word": words, "weight": weight})
df = df[df["word"] != "X"]
df = df.transpose()
df.columns = df.loc['word']
df = df.drop('word')
df = df.convert_objects(convert_numeric=True)

pic = sns.heatmap(df, cmap="Blues", square=True, yticklabels=False, cbar=False, linewidths=0.5)
plt.xticks(rotation=45)
plt.savefig('img/visualization.png', transparent=True)
plt.show()
plt.close()
