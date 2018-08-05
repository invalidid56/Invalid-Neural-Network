import tensorflow as tf

a = tf.placeholder(tf.float32)

b = tf.add(a, 3.)

b = tf.add(b, 3.)

with tf.Session() as sess:
    print(sess.run(b, feed_dict={a: 2.}))