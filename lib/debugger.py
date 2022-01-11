from tensorflow.keras.models import load_model

class Debugger:
    def __init__(self, model_path):
        self.model = load_model(model_path)

        self.input_layer = self._get_layer(index = 0)


    def _get_layer(self, index):
        return self.model.get_layer(index = index)

    def run(self):
        self.model.summary()

