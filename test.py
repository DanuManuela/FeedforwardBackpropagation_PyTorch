import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

net = network.Network([784, 30, 10])
net.SGD(training_data, 4, 3, test_data=test_data)
