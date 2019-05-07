import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn

from attention import attention
from utils import get_sentiment_dataset, load_pretrained_embed


class RnnAttention(object):
    def __init__(self):
        self.embed_size = 300
        self.hidden_size = 150
        self.seq_len = 40
        self.attention_size = 100
        self.dropout_rate = 0.5
        self.num_classes = 2
        self.batch_size = 32
        self.epoch_num = 3
        self.model_path = "./model/model"

    def get_data(self):
        train_data, test_data, self.word2idx, self.n_train, self.n_test = get_sentiment_dataset("data/sentiment.csv",
                                                                                                batch_size=self.batch_size,
                                                                                                seq_len=self.seq_len)
        iterator = tf.data.Iterator.from_structure(train_data.output_types,
                                                   train_data.output_shapes)
        self.text, self.label = iterator.get_next()

        self.train_init = iterator.make_initializer(train_data)
        self.test_init = iterator.make_initializer(test_data)

    def model(self):
        voc_size = len(self.word2idx) + 1
        pretrained_embed = load_pretrained_embed("CBOW_iter15_2017-2018.bin", self.embed_size, self.word2idx)
        embed_matrix = tf.get_variable(name='embedding_matrix',
                                       shape=[voc_size, self.embed_size],
                                       initializer=tf.constant_initializer(pretrained_embed),
                                       dtype=tf.float32)
        embed = tf.nn.embedding_lookup(embed_matrix, self.text)

        # RNN layer
        (fw_outputs, bw_outputs), _ = bi_rnn(GRUCell(self.hidden_size), GRUCell(self.hidden_size), inputs=embed,
                                             dtype=tf.float32)
        rnn_outputs = tf.concat((fw_outputs, bw_outputs), axis=2)
        # Attention layer
        attention_output, self.alpha = attention(rnn_outputs, self.attention_size)
        sentence_vector = tf.layers.dropout(attention_output, self.dropout_rate)

        self.logits = tf.layers.dense(inputs=sentence_vector, units=self.num_classes, name='logits')

    def loss(self):
        self.one_hot_labels = tf.one_hot(self.label, depth=self.num_classes, dtype=tf.float32)
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.one_hot_labels)
        self.loss = tf.reduce_mean(entropy, name='loss')

    def optimize(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)

    def predict(self):
        preds = tf.nn.softmax(self.logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.one_hot_labels, 1))
        self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.histogram('histogram loss', self.loss)
        self.merged = tf.summary.merge_all()

    def build(self):
        self.get_data()
        self.model()
        self.loss()
        self.optimize()
        self.predict()
        self.summary()

    def train(self):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('./graphs/rnn_attention', sess.graph)
            # train the model
            num_batches = self.n_train // self.batch_size  # steps per epoch
            for epoch in range(self.epoch_num):
                sess.run(self.train_init)
                total_loss = 0
                n_batches = 0
                try:
                    while True:
                        _, l, summary = sess.run([self.optimizer, self.loss, self.merged])
                        writer.add_summary(summary, n_batches + num_batches * epoch)
                        total_loss += l
                        n_batches += 1
                except tf.errors.OutOfRangeError:
                    pass
                print('Average loss epoch {0}: {1}'.format(epoch, total_loss / n_batches))
            # test
            sess.run(self.test_init)
            total_correct_preds = 0
            try:
                while True:
                    accuracy_batch = sess.run(self.accuracy)
                    total_correct_preds += accuracy_batch
            except tf.errors.OutOfRangeError:
                pass
            print('Accuracy {0}'.format(total_correct_preds / self.n_test))
            writer.close()
            saver.save(sess, self.model_path)
            print("Run 'tensorboard --logdir=./graphs/rnn_attention' to checkout tensorboard logs.")


if __name__ == '__main__':
    model = RnnAttention()
    model.build()
    model.train()
