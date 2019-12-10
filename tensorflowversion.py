import tensorflow as tf
import matplotlib.pyplot as plot
 
# load mnist data into appropriate variables
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()

# normalize the data to 0 - 1
tf.keras.utils.normalize(train_img, axis=1)
tf.keras.utils.normalize(test_img, axis=1)
# plot.imshow(train_img[0])
# plot.show()

# initialize model
network = tf.keras.models.Sequential()
network.add(tf.keras.layers.Flatten())

# layers and nodes were chosen based off the best tested values from HWDigitRecognizer.py
network.add(tf.keras.layers.Dense(150, activation=tf.nn.sigmoid))
network.add(tf.keras.layers.Dense(120, activation=tf.nn.sigmoid))

# output layer uses softmax to convert to probability distribution
network.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# optimzer = stochastic gradient descent
# sparse is used since labels are not one hot encoded
network.compile(optimzer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# give this network the same amount of training data as the other
network.fit(train_img, train_labels, epochs=1)

