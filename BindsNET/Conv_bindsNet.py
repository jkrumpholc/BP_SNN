import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.analysis.plotting import plot_spikes
import matplotlib.pyplot as plt

# Parameters
time = 100  # Simulation time in ms
dt = 1.0  # Time step
batch_size = 1  # One image at a time
n_output = 10  # Number of neurons in output layer (for classification)

# Data Transformation and Loading
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)


# Define Convolutional Layers Using PyTorch (pre-processing)
class ConvPreprocessing(nn.Module):
    def __init__(self):
        super(ConvPreprocessing, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        return x


conv_model = ConvPreprocessing()

# Initialize BindsNET network
network = Network(dt=dt)

# Define Poisson Encoder
encoder = PoissonEncoder(time=time)

# Create input LIF Layer (Post-processed features)
input_size = 32 * 4 * 4  # Adjusted to flattened conv2 output shape
input_layer = LIFNodes(n=input_size)
network.add_layer(input_layer, name='input_layer')

# Fully Connected Output Layer for classification
output_layer = LIFNodes(n=n_output)
network.add_layer(output_layer, name='output_layer')

# Connection between input layer and output layer
fc_connection = Connection(source=input_layer, target=output_layer)
network.add_connection(fc_connection, source='input_layer', target='output_layer')

# Add monitors to record spikes
input_monitor = Monitor(network.layers['input_layer'], state_vars=['s'], time=time)
output_monitor = Monitor(network.layers['output_layer'], state_vars=['s'], time=time)
network.add_monitor(input_monitor, name='input_spikes')
network.add_monitor(output_monitor, name='output_spikes')

# Run the network on a single batch (for demonstration purposes)
for batch_idx, (images, labels) in enumerate(trainloader):
    print(f"Processing batch {batch_idx + 1}/{len(trainloader)}")

    # Forward pass through pre-processing conv layers
    with torch.no_grad():
        conv_features = conv_model(images)

    # Encode conv features with Poisson encoding
    encoded_features = encoder(conv_features.view(-1))

    # Simulate the network
    network.run(inputs={'input_layer': encoded_features}, time=time)

    # Retrieve spikes
    input_spikes = input_monitor.get('s')
    output_spikes = output_monitor.get('s')

    # Plot the spikes
    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plot_spikes({'input_spikes': input_spikes}, time=(0, time))
    plt.title('Input Layer Spikes (Post Conv)')

    plt.subplot(212)
    plot_spikes({'output_spikes': output_spikes}, time=(0, time))
    plt.title('Output Layer Spikes')
    plt.tight_layout()
    plt.show()

    # Stop after one batch for demonstration
    break

# Reset the network state for next simulation
network.reset_state_variables()
