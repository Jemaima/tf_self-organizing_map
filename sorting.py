from SOM import SOMNetwork as SOM
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

initial_dim = 6
som = SOM(3, initial_dim, sigma=initial_dim / 2)
x_test = np.random.uniform(0, 1, (initial_dim * initial_dim, 3))

train_op = som.train_op()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    n_iters = 500
    for j in range(n_iters):
        np.random.shuffle(x_test)
        for i, color_data in enumerate(x_test):
            if (j * n_iters + i) % 100 == 0:
                print('iter:', j * n_iters + i)
            a = sess.run(train_op, feed_dict={som.x: color_data, som.n_iter: j * n_iters + i})
            if np.count_nonzero(a[1] > 0.01) == 1:
                print('self.sigma <= self.min_sigma')
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow((1 + np.reshape(x_test, [initial_dim, initial_dim, 3])) / 2)
    plt.subplot(1, 2, 2)
    plt.imshow((1 + np.reshape(sess.run(som.w), [som.dim.eval(), som.dim.eval(), 3])) / 2)

    plt.figure()
    plt.imshow((np.reshape(a[1], [som.dim.eval(), som.dim.eval()])))
    plt.show(block=True)
