import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.encoding import PoissonEncoder
from bindsnet.analysis.plotting import plot_spikes
import matplotlib.pyplot as plt

# Parameters
n_input = 28 * 28  # For MNIST, each image is 28x28 pixels
n_output = 100  # Number of output neurons
time = 250  # Simulation time (ms)
dt = 1.0  # Time step
batch_size = 1  # Process one image at a time (can increase this for faster processing)

# Define transformations for the MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

# Load the MNIST dataset
trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

# Initialize the network
network = Network(dt=dt)

# Define input and output layers
input_layer = Input(n=n_input)
output_layer = LIFNodes(n=n_output)

# Add layers to the network
network.add_layer(layer=input_layer, name='input')
network.add_layer(layer=output_layer, name='output')

# Connect input to output layer
connection = Connection(source=input_layer, target=output_layer)
network.add_connection(connection, source='input', target='output')

# Add spike monitors
input_monitor = Monitor(input_layer, state_vars=['s'])
output_monitor = Monitor(output_layer, state_vars=['s'])
network.add_monitor(input_monitor, name='input')
network.add_monitor(output_monitor, name='output')

# Poisson Encoder
encoder = PoissonEncoder(time=time)

# Loop over the entire MNIST dataset
for batch_idx, (labels, images) in enumerate(trainloader):
    print(f"Processing batch {batch_idx + 1}/{len(trainloader)}")

    # Flatten the image into a 1D array (28x28 = 784)
    mnist_image = images.view(-1)

    # Encode the MNIST image into Poisson spike trains
    encoded_image = encoder(mnist_image)

    # Run the network
    spikes = {'input': encoded_image}
    network.run(inputs=spikes, time=time)

    # Retrieve and plot the spikes for each batch (optional: can be removed for faster processing)
    input_spikes = input_monitor.get('s')
    output_spikes = output_monitor.get('s')

    # Plotting input and output spikes for this batch
    plt.figure(figsize=(12, 6))
    plt.subplot(211)
    plot_spikes({'input': input_spikes}, time=(0, time))
    plt.title(f'Input Layer Spikes (Batch {batch_idx + 1})')

    plt.subplot(212)
    plot_spikes({'output': output_spikes}, time=(0, time))
    plt.title(f'Output Layer Spikes (Batch {batch_idx + 1})')
    plt.tight_layout()
    plt.show()

    # Reset the network to clear states for the next image
    network.reset_state_variables()

    # For testing: break the loop early after processing a few batches (remove this to process all data)
    if batch_idx == 9:  # Change the number to control how many batches to process
        break