import  mnist_loader
import  network
import imageio
import numpy as np
from numpy  import array
import json
np.seterr(all='warn')

net = network.Network([784, 100, 10])
json_data = open("nn784-100-10_50-epochs.json").read()
nn_body = json.loads(json_data)

sizes = nn_body["sizes"]
num_layers = nn_body["num_layers"]

parsed_biases = []
for layer in nn_body["biases"]:
    parsed_biases.append(np.asarray(layer, dtype=np.float64).reshape(len(layer), 1))

unconverted_weights = []
for layer in nn_body["weights"]:
    layer_weights = []
    for w in layer:
        layer_weights.extend(w)
    unconverted_weights.append(layer_weights)

parsed_weights = []
for i, layer in enumerate(unconverted_weights):
    parsed_weights.append(np.asarray(layer, dtype=np.float64).reshape(net.sizes[i+1], net.sizes[i]))

net.sizes = sizes
net.num_layers = num_layers
net.biases = parsed_biases
net.weights = parsed_weights

image = imageio.imread('images/testing/4/24.png')
test_image = np.reshape(image, (784, 1))
result = net.feedforward(test_image)
print(np.argmax(result))

# weights = []
# for l in len(net.weights):
#     if isinstance(item, np.ndarray):
#         weights.append(item.tolist())
#     else:
#         weights.append(item)
#
# json.dump({'biases': biases}, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

