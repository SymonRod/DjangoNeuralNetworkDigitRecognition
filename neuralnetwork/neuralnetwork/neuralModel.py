import tensorflow as tf
import numpy as np

class nn():
    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (training_data, training_label), (test_data,test_label) = mnist.load_data()
        training_data, test_data = training_data/255,test_data/255
        try:
            self.model = tf.keras.models.load_model('./model/neuralNetwork')
        except:
            
            self.model = tf.keras.Sequential([
                tf.keras.layers.Flatten(input_shape=[28,28]),
                tf.keras.layers.Dense(128,activation=tf.nn.relu),
                tf.keras.layers.Dense(10,activation=tf.nn.softmax)
            ])
            self.model.compile(optimizer=tf.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",metrics=['accuracy']
            )
            self.model.fit(training_data,training_label,epochs=5)
            self.model.evaluate(test_data,test_label)
            self.model.save('./model/neuralNetwork')

        # print(test_data)

        # np.set_printoptions(suppress=True)
        # self.model.evaluate(test_data,test_label)
        # prediction = self.model.predict(test_data)
        # print(prediction[0])
        # print(test_label[0])

    def predict(self,array):
        return self.model.predict(array)