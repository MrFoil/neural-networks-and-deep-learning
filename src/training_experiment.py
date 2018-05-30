import  mnist_loader
import  network
import imageio

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network.Network([784, 100, 10])
test_net = network.Network([784, 100, 10])
net.SGD(training_data, 50, 10, 3.0, test_data=test_data)
# net.save_body_to_file("nn784-100-10_50-epochs.json")

# image = imageio.imread('images/testing/3/2109.png')
# print(image.shape)
# test_image = np.reshape(image, (784, 1))
# result = net.feedforward(test_image)
# print(result.tolist())
