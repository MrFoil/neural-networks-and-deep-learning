import  mnist_loader
import  network
import imageio

def initexperiment(sizes, epochs, eta):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network(sizes)
    net.SGD(training_data, epochs, 10, eta, test_data=test_data)

def read_input():
    print("Enter the sizes of layers in unsigned integers (e.g. '784, 100, 10'): ")
    input_sizes = input()
    sizes = [int(s.strip()) for s in input_sizes.split(',')]

    print("Enter the number of training cycles to go through in unsigned integers (e.g. '50'): ")
    input_epochs = input()
    epochs = int(input_epochs)

    print("Enter the the learning rate or so called 'eta' as a float number (e.g. '2.9'): ")
    input_eta = input()
    eta = float(input_eta)

    initexperiment(sizes, epochs, eta)

# net.save_body_to_file("nn784-100-10_50-epochs.json")
# image = imageio.imread('images/testing/3/2109.png')
# print(image.shape)
# test_image = np.reshape(image, (784, 1))
# result = net.feedforward(test_image)
# print(result.tolist())


if __name__ == "__main__":
    read_input()
