import network
import imageio
import sys
import numpy as np


def main(network_body_path, file_path):
    np.seterr(all='ignore')
    net = network.Network(network_body_path)  # "nn784-100-10_50-epochs.json"

    image = imageio.imread(file_path)  # images/testing/5/1940.png
    test_image = np.reshape(image, (784, 1))
    result = net.feedforward(test_image)
    print("THE ANSWER IS: {0}".format(np.argmax(result)))


if __name__ == "__main__":
    first = str(sys.argv[1])
    second = str(sys.argv[2])
    main(first, second)