from lib.debugger import SimpleDebugger
from lib.utils import limit_gpu

def example_with_image():
    debugger = SimpleDebugger(model_path = "/home/musa/python_pro/tf2_cookbooks/one/search.h5")
    debugger.load_model_with_index(to = 6)
    debugger.run_on(image_path = "/home/musa/Downloads/bird.jpg")


def debug_whole_model():
    debugger = SimpleDebugger(model_path = "/home/musa/python_pro/tf2_cookbooks/one/search.h5")
    debugger.debug_whole_model(image_path = "/home/musa/Downloads/bird.jpg", save_dir_path = "test")



def main():
    limit_gpu()
    debug_whole_model()


if __name__ == "__main__":
    main()
