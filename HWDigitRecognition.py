from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import pickle




def main():
    l = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    d = NeuralNetwork(784, [300, 300], 10, l)

    # with open("pickled/pickled_mnist" + str(0) + ".plk", "br") as fh:
    #     data = pickle.load(fh)

    # train_imgs = data[0]
    # tain_labels = data[1]

    # print(train_imgs)
    # print(tain_labels)

    targets = np.asarray([
               [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
               [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

    print(targets)

    # train = [] 
    # row = 0
    # last = 0
    # batch = 0
    # for line in open("mnist_train.csv"):
    #     train.append(np.fromstring(line, sep=","))
    #     if row == (60 + last):
    #         last = row
    #         fac = 0.99 / 255
    #         train_imgs = np.asfarray(train) 
    #         train_labels = np.asfarray(train)

    #         train_imgs = train_imgs[: , 1:] * fac + 0.01
    #         train_labels = train_labels[: , :1]
    #         with open("pickled/pickled_mnist" + str(batch) + ".plk", "bw") as fh:
    #             data = (train_imgs, train_labels)
    #             pickle.dump(data, fh)
    #         batch += 1
    #         train = []
    #     row += 1

    

    # img = train_imgs[1].reshape((28,28))
    # plt.imshow(img, cmap="Greys")
    # plt.show()


if __name__ == '__main__':
    main()