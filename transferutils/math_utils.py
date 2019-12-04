import tensorflow as tf


def get_features_loss(noise_features, features_target):
    return tf.reduce_mean(tf.square(noise_features - features_target))


def gram_matrix(tensor):
    tensor = tf.reshape(tensor, [-1, int(tensor.shape[-1])])
    filters = int(tf.shape(tensor)[0])
    gram = tf.matmul(tensor, tensor, transpose_a=True)
    return gram / tf.cast(filters, tf.float32)


def get_style_loss(noise_features, target_features):
    noise_gram = gram_matrix(noise_features)
    target_gram = gram_matrix(target_features)
    return tf.reduce_sum(tf.square(noise_gram - target_gram)) / tf.cast(4, tf.float32)