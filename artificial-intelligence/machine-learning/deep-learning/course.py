"""https://web.stanford.edu/class/cs20si/"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = "../../../../data/fire_theft.xls"

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input
# X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# Step 3: create variables: weights_1, weights_2, bias.
# All are initialized to 0
w = tf.Variable(0.0, name="weights_1")
u = tf.Variable(0.0, name="weights_2")
b = tf.Variable(0.0, name="bias")

# Step 4: predict Y (number of theft) from the number of fire
Y_predicted = X * X * w + X * u + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name="loss")

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001) \
    .minimize(loss)
with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer())

    # Step 8: train the model
    for i in range(100):  # run 10 epochs
        for x, y in data:
            # Session runs train_op to minimize loss
            _, res_loss = sess.run((optimizer, loss), feed_dict={X: x, Y: y})
        print("[Epoch {}] medium loss = {}".format(i, res_loss / n_samples))

    # Step 9: output the values of w, u and b
    w, u, b = sess.run([w, u, b])
    print("w: {}, u: {}, b: {}".format(w, u, b))
    writer = tf.summary.FileWriter('./log_tensorboard', sess.graph)
# close the writer when you’re done using it
writer.close()

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * X * w + X * u + b, 'r', label='Predicted data')
plt.legend()
plt.show()
