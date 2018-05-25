import  mnist_loader
import  network
import imageio
import numpy as np

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
net.SGD(training_data, 10, 10, 3.0, test_data=test_data)
#net.save_body_to_file("output_nn_150_5.json")

image = imageio.imread('images/testing/3/2109.png')
print(image.shape)
test_image = np.reshape(image, (784, 1))
result = net.feedforward(test_image)
print(result.tolist())


# weights = []
# for l in len(net.weights):
#     if isinstance(item, np.ndarray):
#         weights.append(item.tolist())
#     else:
#         weights.append(item)
#
# json.dump({'biases': biases}, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

