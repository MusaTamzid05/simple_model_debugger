import tensorflow as tf
import cv2

def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit = 2048)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)



def show_image(image, window_name = "Window"):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

