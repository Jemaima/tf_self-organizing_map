from SOM import SOMNetwork as SOM
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

som = SOM(3, 10, sigma=20)
x_test = np.random.uniform(0, 1, (100000, 3))

train_op = som.train_op()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for i, color_data in enumerate(x_test):
        if i % 1000 == 0:
            print('iter:', i)
        a = sess.run(train_op, feed_dict={som.x: color_data, som.n_iter: i})
    plt.figure()
    plt.imshow( np.reshape(sess.run(som.w), [som.dim.eval(), som.dim.eval(), 3]))
    plt.show(block=False)
    plt.figure()
    plt.imshow((np.reshape(a[1], [som.dim.eval(), som.dim.eval()])))
    plt.show()
