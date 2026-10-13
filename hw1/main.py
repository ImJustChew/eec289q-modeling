import os
import struct
from array import array
from datetime import datetime
from os.path import join
from pickletools import pylong

import matplotlib.pyplot as plt
import numpy as np


# https://www.kaggle.com/code/hojjatk/read-mnist-dataset
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test)


input_path = 'dataset'
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

print("Training data shape:", np.array(x_train).shape)

# keep only 0, 5, 8 digits
x_train, y_train = zip(*[(x, y) for x, y in zip(x_train, y_train) if y in [0, 5, 8]])
x_test, y_test = zip(*[(x, y) for x, y in zip(x_test, y_test) if y in [0, 5, 8]])

# goal: develop a 0/5/8 classifier using simple linear perceptron model.
# we develope 3 simple features to classify the digits.
# preprocess; normalize the data to [0,1] range within its own min max

# normalize the images
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

## create the new features for training and testing data
def calculate_features(images):
    features = []
    for img in images:
        center_3x3 = np.mean(img[13:16, 13:16])
        avg_intensity = np.mean(img)
        mid_row_mass = np.mean(img[14, :])
        features.append([center_3x3, avg_intensity, mid_row_mass])
    return np.array(features)

x_test_features = calculate_features(x_test)
x_train_features = calculate_features(x_train)

# plot the 3 feature map to visualize the features for the entire dataset, with the color of dots representing the labels.
plt.figure(figsize=(10, 10))
plt.scatter(x_train_features[:, 0], x_train_features[:, 1], c=['red' if label == 0 else 'green' if label == 5 else 'blue' for label in y_train], alpha=0.5)
plt.title('Feature Map for Digits 0, 5, 8')
plt.xlabel('Center Intensity')
plt.ylabel('Average Intensity')
plt.savefig('feature_map_center_avg.png')

plt.figure(figsize=(10, 10))
plt.scatter(x_train_features[:, 0], x_train_features[:, 2], c=['red' if label == 0 else 'green' if label == 5 else 'blue' for label in y_train], alpha=0.5)
plt.title('Feature Map for Digits 0, 5, 8')
plt.xlabel('Center Intensity')
plt.ylabel('Mid Row Mass')
plt.savefig('feature_map_center_midrow.png')

plt.figure(figsize=(10, 10))
plt.scatter(x_train_features[:, 1], x_train_features[:, 2], c=['red' if label == 0 else 'green' if label == 5 else 'blue' for label in y_train], alpha=0.5)
plt.title('Feature Map for Digits 0, 5, 8')
plt.xlabel('Average Intensity')
plt.ylabel('Mid Row Mass')
plt.savefig('feature_map_avg_midrow.png')

np.random.seed(42)

# 5-fold cross validation, split into 5 folds along with their y labels
folds_x = np.array_split(x_train_features, 5)
folds_y = np.array_split(y_train, 5)

# develop a classifier using the simple linear perceptron model
# +1 for positive, -1 for negative
# we build 3 classifiers for 0, 5, and 8. Each classifier is trained to distinguish between the target digit and the other digits.

class Perceptron:
    def __init__(self, input_shape) -> None:
        # we pad the bias term to the weights, so we don't need to add it separately in the forward pass.
        self.weights = np.random.randn(input_shape + 1)
        self.loss_history = []
        self.validation_loss_history = []

    def predict(self, x):
        # we pad the bias term to the input, so we don't need to add it separately in the forward pass.
        x_n = np.append(x, 1)
        if np.dot(self.weights, x_n) >= 0:
            return 1
        else:
            return -1

    def train(self, x, y, x_valid, y_valid, epochs=1):
        for epoch in range(epochs):
            for i in range(len(x)):
                y_pred = self.predict(x[i])
                if not y_pred == y[i]:
                    self.weights += y[i] * np.append(x[i], 1)  # update the weights with the bias term
                else:
                    continue
            # we define the loss as the number of misclassified samples in each epoch.
            loss = np.sum([1 for i in range(len(x)) if self.predict(x[i]) != y[i]])
            self.loss_history.append(loss/len(x))
            validation_loss = np.sum([1 for i in range(len(x_valid)) if self.predict(x_valid[i]) != y_valid[i]])
            self.validation_loss_history.append(validation_loss/len(x_valid))

# we need to save all the graphs and results systematically
# we tag the current timestamp and save results to the folder
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(current_time, exist_ok=True)

for digit in [0, 5, 8]:
    print(f"Training classifier for digit {digit}...")
    accuracies = []
    train_losses = []
    valid_losses = []

    # create labels fpr ths classifier
    folds_y_digit = [np.array([1 if label == digit else -1 for label in fold]) for fold in folds_y]
    y_test_digit = np.array([1 if label == digit else -1 for label in y_test])

    # show the feature map to visualize the features for the current digit of the entire dataset, wiuth the color of dots representing the labels.
    plt.figure(figsize=(10, 10))
    plt.scatter(x_train_features[:, 0], x_train_features[:, 1], c=['red' if label == digit else 'blue' for label in y_train], alpha=0.5)
    plt.title(f'Feature Map for Digit {digit}')
    plt.xlabel('Center Intensity')
    plt.ylabel('Average Intensity')
    plt.savefig(f'{current_time}/feature_map_digit_{digit}_center_avg.png')

    plt.figure(figsize=(10, 10))
    plt.scatter(x_train_features[:, 0], x_train_features[:, 2], c=['red' if label == digit else 'blue' for label in y_train], alpha=0.5)
    plt.title(f'Feature Map for Digit {digit}')
    plt.xlabel('Center Intensity')
    plt.ylabel('Mid Row Mass')
    plt.savefig(f'{current_time}/feature_map_digit_{digit}_center_midrow.png')

    plt.figure(figsize=(10, 10))
    plt.scatter(x_train_features[:, 1], x_train_features[:, 2], c=['red' if label == digit else 'blue' for label in y_train], alpha=0.5)
    plt.title(f'Feature Map for Digit {digit}')
    plt.xlabel('Average Intensity')
    plt.ylabel('Mid Row Mass')
    plt.savefig(f'{current_time}/feature_map_digit_{digit}_avg_midrow.png')

    for fold in range(5):
        print(f"Fold {fold + 1}...")
        # we use the current fold as the validation set, and the rest as the training set.
        x_valid = folds_x[fold]

        y_valid_digit = folds_y_digit[fold]
        x_train_fold = np.concatenate([folds_x[i] for i in range(5) if i != fold])
        y_train_digit = np.concatenate([folds_y_digit[i] for i in range(5) if i != fold])

        perceptron = Perceptron(input_shape=x_train_fold.shape[1])
        perceptron.train(x_train_fold, y_train_digit, x_valid, y_valid_digit, epochs=1000)

        # output single digit classification accuracy on the test set
        test_accuracy = np.sum([1 for i in range(len(x_test_features)) if perceptron.predict(x_test_features[i]) == y_test_digit[i]]) / len(x_test_features)
        print(f"Test accuracy for digit {digit}: {test_accuracy * 100:.2f}%")

        # save for this fold
        accuracies.append(test_accuracy)
        train_losses.append(perceptron.loss_history)
        valid_losses.append(perceptron.validation_loss_history)


    # average the accuracies and losses across the folds
    avg_accuracy = np.mean(accuracies)
    avg_train_loss = np.mean(train_losses, axis=0)
    avg_valid_loss = np.mean(valid_losses, axis=0)
    print(f"Average test accuracy for digit {digit}: {avg_accuracy * 100:.2f}%")

    # plot the average training and validation loss.
    plt.figure(figsize=(10, 5))
    plt.plot(avg_train_loss, label='Average Training Loss')
    plt.plot(avg_valid_loss, label='Average Validation Loss')
    plt.title(f'Loss History for Digit {digit}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Misclassified Samples)')
    plt.legend()
    plt.savefig(f'{current_time}/loss_history_digit_{digit}_average.png')

    # save results to a text file
    with open(f'{current_time}/results_digit_{digit}.txt', 'w') as f:
        f.write(f"Average test accuracy for digit {digit}: {avg_accuracy * 100:.2f}%\n")
        f.write(f"Training loss history: {avg_train_loss.tolist()}\n")
        f.write(f"Validation loss history: {avg_valid_loss.tolist()}\n")
