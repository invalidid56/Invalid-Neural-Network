import tensorflow as tf

a = tf.Variable([[1, 2, 3]])
if (tf.rank(a) > 5) is not None:
    print('f')
