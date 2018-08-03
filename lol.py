import tensorflow as tf

a = 5.

b = tf.add(a, 3.)

with tf.Session() as sess:
    print(sess.run(b))