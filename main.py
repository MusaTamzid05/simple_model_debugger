from lib.debugger import Debugger
from lib.utils import limit_gpu

def main():
    limit_gpu()
    debugger = Debugger(model_path = "/home/musa/python_pro/tf2_cookbooks/one/search.h5")
    debugger.load_model_with_index(from_= 0, to = 5)
    debugger.run_on(image_path = "/home/musa/Downloads/bird.jpg")

if __name__ == "__main__":
    main()
