from tensorflow import keras
import os

class ImageDebuggerCallback(keras.callbacks.Callback):

    def __init__(self, image, save_dir_path):
        keras.callbacks.Callback.__init__(self)
        self.image = image

        if os.path.exists(save_dir_path) == False:
            os.mkdir(save_dir_path)

        self.save_dir_path = save_dir_path


    def on_epoch_end(self, epochs, logs = None):
        print("\nThis is epoch end.")
        print(self.image.shape)

