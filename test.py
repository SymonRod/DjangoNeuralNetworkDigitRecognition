
import  tensorflow as tf
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist
(training_data, training_label), (test_data,test_label) = mnist.load_data()

for i in range(20):
    # reshape to 2d
    mat = np.reshape(training_data[i],(28,28))

    # Creates PIL image
    img = Image.fromarray(np.uint8(mat * 255) , 'L')
    img.show()