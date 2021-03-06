from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras import models

import cv2
import os
import numpy as np

from lib.utils import show_image
from matplotlib import pyplot as plt


class SimpleDebugger:
    def __init__(self, model):

        if type(model) == str:
            self.model = load_model(model)
        else:
            self.model = model

        self.input_shape = self.model.input.shape


    def load_model_with_index(self, to):
        self._generate_model_with_index(to = to)

    def _get_layer(self, index):
        return self.model.get_layer(index = index)

    def _generate_model_with_index(self, to):
        layer_outputs = [layer.output for layer in self.model.layers[:to]]
        self.current_model = models.Model(inputs = self.model.input, outputs = layer_outputs)


    def _load_image_for_process(self, image_path):
        image = cv2.imread(image_path)

        if image is None:
            raise RuntimeError(f"{image_path} does not exist")

        image_channel = self.input_shape[3]

        if image_channel == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image



    def _process_image(self, image):

        if type(image) == str:
            image = self._load_image_for_process(image_path = image)

        image = cv2.resize(image, (self.input_shape[1], self.input_shape[2]))
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


    def _load_whole_model(self):
        total_layers = len(self.model.layers)
        self._generate_model_with_index(to = total_layers )
        self.current_model.summary()


    def _save_whole_result(self, results, save_dir_path):
        if os.path.isdir(save_dir_path) == False:
            os.mkdir(save_dir_path)

        minimum_channel = 4
        required_shape = 4

        for index, result in enumerate(results):
            plt.clf()
            layer_name = self.current_model.layers[index].name
            result_image = results[index]
            image_name = f"{index}_{layer_name}.jpg"


            if len(result_image.shape) != required_shape:
                print(f"Avoding {image_name} because of incompatible shape")
                continue

            if result_image.shape[3] > minimum_channel:
                index = 4
            else:
                index = 0

            plt.matshow(result_image[0, :, :, index], cmap = "viridis")
            image_path = os.path.join(save_dir_path, image_name)
            plt.savefig(image_path)





    def debug_whole_model(self, image, save_dir_path):
        self._load_whole_model()
        image = self._process_image(image = image)


        results = self.current_model.predict(image)
        self._save_whole_result(results = results, save_dir_path = save_dir_path)
