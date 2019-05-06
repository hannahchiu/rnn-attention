import tensorflow as tf


def attention(inputs, attention_size):
    hidden_size = inputs.shape[2].value

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    u = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    uu = tf.tensordot(u, u_omega, axes=1)
    alpha = tf.nn.softmax(uu)

    s = tf.reduce_sum(inputs * tf.expand_dims(alpha, -1), 1)
    return s
