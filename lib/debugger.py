from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras import models

import cv2
import numpy as np

from lib.utils import show_image
from matplotlib import pyplot as plt


class Debugger:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self._init_input_layer()


    def load_model_with_index(self, to):
        self._generate_model_with_index(to = to)


    def _init_input_layer(self):
        self.input_layer = self._get_layer(index = 0)
        input_shape = self.input_layer.input_shape[0]
        self.size = input_shape[1], input_shape[2]
        self.channel = input_shape[3]


    def _get_layer(self, index):
        return self.model.get_layer(index = index)

    def _generate_model_with_index(self, to):
        layer_outputs = [layer.output for layer in self.model.layers[:to]]
        self.current_model = models.Model(inputs = self.model.input, outputs = layer_outputs)


    def _process_image(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"{image_path} does not exist")

        if self.channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(image, (self.size[0], self.size[1]))
        image = np.expand_dims(image, -1)
        image = np.expand_dims(image, 0)

        return image

    def _get_best_diemnention(self, size):
        current_size = 2
        last_size = None

        while (current_size * current_size) < size:
            last_size = current_size
            current_size *= 2

        return last_size



    def _resize_result(self, result):
        result = result.reshape((1, result.shape[0] * result.shape[1] * result.shape[2]))
        best_size = self._get_best_diemnention(size = result.shape[1])

        result = cv2.resize(result, (best_size, best_size))

        result = np.expand_dims(result, -1)

        return result


    def run_on(self, image_path, save_path = None):

        self.model.summary()

        if self.current_model is None:
            raise RuntimeError("Current model is not set yet.")

        self.current_model.summary()

        image = self._process_image(image_path = image_path)
        result = self.current_model.predict(image)

        layer = result[len(result) - 1]

        plt.matshow(layer[0, :, :, 4], cmap = "viridis")


        if save_path is not None:
            plt.savefig(save_path)
        plt.show()





