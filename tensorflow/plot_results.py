import matplotlib.pyplot as plt
import numpy as np


def plot_results(nrows, ncols, predictions, test_labels, test_images):
    '''
    Args
    ----
    nrows: int
            Number of rows to be displayed for test
    ncols: int
            Number of cols to be displayed for test
    predictions: list or numpy array
            Predictions for each test sample. For a test with 3 classes & 2 test images - shape will be (2, 3)
    test_labels: list
            Actual labels
    test_images: numpy array
            Test images. For mnist dataset, shape will be (10000, 28, 28)
    '''
    num_images = nrows*ncols
    plt.figure(figsize=(2*2*ncols, 2*nrows))
    for i in range(num_images):
        plt.subplot(nrows, 2*ncols, 2*i+1)
        plot_image(i, predictions[i], test_labels, test_images)
        plt.subplot(nrows, 2*ncols, 2*i+2)
        plot_value_array(i, predictions[i], test_labels)
    plt.tight_layout()
    plt.show()


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
      color = 'blue'
    else:
      color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
