import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
c = tf.Variable(6.)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    c = c+5.
    save_path = saver.save(sess, 'C:\Temp\model.ckpt')


with tf.Session() as sess:
    saver.restore(sess, 'C:\Temp\model.ckpt')
    d = sess.run(c)

print(d)
