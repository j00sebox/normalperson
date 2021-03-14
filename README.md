# normalperson

## Description

A neural network to recognize handwritten digits made with the basicNN. Trained using the MNIST dataset. Composed of 2 hidden layers with 300 and 100 HU respectively.

The MNIST training and test sets come in the form of csv files. In order to make the training and testing files faster and more efficient the data is pickled before hand and the fucntions are expecting it to be in this format. The data is pickled into batches of 60 samples.

## Testing

The testing function is a bit unconventional. It gives the user a prompt of what testing batch to select, then what specific sample in the batch to test. Once that is done the model will predict the number and display it's predict in the console, then the actual image will be shown to the user. This a less practical way to test a model and more of a way to visually show it is working. 

![Guess_1](/screenshots/guess1.PNG)

![Pic_1](/screenshots/pic1.PNG)

![Guess_2](/screenshots/guess2.PNG)

![Pic_2](/screenshots/pic2.PNG)

There is also an option to calculate the accuracy of the model over n randomly chosen samples. This outputs a percentage of the samples it correctly guessed and is a much better indicator of how the model is performing.

Accuracy of model over 5000 random samples:

![Acc](/screenshots/acc.PNG)

## How to Use

| Command | Description |
| --- | --- |
| -T | Train model on pickled dataset |
| -t | Test model on pickled test set |
| -a | Integer. Calculate accuracy based on a certain number of test samples. |
| -p | Pickle test and training data. |

## tensorflowversion.py

I have also included a version of this project that use TensorFlow. It does the same thing but with less code. It only has argument options for testing and training.
