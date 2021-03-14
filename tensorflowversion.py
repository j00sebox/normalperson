import tensorflow as tf
import matplotlib.pyplot as plot
import random 
import numpy as np
import argparse

# load mnist data into appropriate variables
(train_img, train_labels), (test_img, test_labels) = tf.keras.datasets.mnist.load_data()

# normalize the data to 0 - 1
tf.keras.utils.normalize(train_img, axis=1)
tf.keras.utils.normalize(test_img, axis=1)

def train():
    # initialize model
    # start adding layers
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Flatten())

    # layers and nodes were chosen based off the best tested values from HWDigitRecognizer.py
    network.add(tf.keras.layers.Dense(300, activation=tf.nn.sigmoid))
    network.add(tf.keras.layers.Dense(100, activation=tf.nn.sigmoid))

    # output layer uses softmax to convert to probability distribution
    network.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    # optimzer = stochastic gradient descent
    # sparse is used since labels are not one hot encoded
    network.compile(optimzer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # give this network the same amount of training data as the other
    network.fit(train_img, train_labels, epochs=1)

    network.save('digitRecognizer.model')

def test():
    # load network from training
    network = tf.keras.models.load_model('digitRecognizer.model')

    # pick a random test case to predict
    r = random.randint(0, len(test_img))
    pred = network.predict([test_img])[r]

    # output the prediction and actual label
    print('Guess: ', np.argmax(pred))
    print('Actual: ', test_labels[r])
    
    # output image
    plot.imshow(test_img[r])
    plot.show()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-T', action='store_true', dest='train', help='Train model on dataset.')
    parser.add_argument('-t', action='store_true', dest='test', help='Enter testing mode.')

     if (args.train):
        train()

    if (args.test):
        test()
 

if __name__ == '__main__':
    main()