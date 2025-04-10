import torch
import matplotlib.pyplot as plt

from bindsnet.network import Network
from bindsnet.network.nodes import Input, DiehlAndCookNodes
from bindsnet.network.topology import Conv2dConnection
from bindsnet.learning import PostPre
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.analysis.plotting import plot_spikes

from bindsnet.evaluation import assign_labels, all_activity
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device config
device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
time = 100
n_classes = 10
input_shape = (1, 28, 28)
conv_filters = 16
conv_kernel = 5
conv_stride = 1
n_epochs = 1
samples_per_epoch = 300

# Build network
network = Network(dt=1.0)
network.to(device)

# Layers
input_layer = Input(n=28 * 28, shape=(1, 28, 28), traces=True, device=device)
conv_layer = DiehlAndCookNodes(n=conv_filters * 24 * 24, shape=(conv_filters, 24, 24), traces=True)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')

# Conv connection (STDP)
conv_conn = Conv2dConnection(
    source=input_layer,
    target=conv_layer,
    kernel_size=conv_kernel,
    stride=conv_stride,
    update_rule=PostPre,
    nu=(1e-4, 1e-2),
    wmin=0.0,
    wmax=1.0,
    norm=0.4 * conv_kernel ** 2
)

network.add_connection(conv_conn, source='X', target='Y')

# Dataset
transform = transforms.ToTensor()
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_dataset = MNIST(PoissonEncoder(time=time, dt=1), None, "./data/MNIST", train=True, download=True, transform=None)
test_dataset = MNIST(PoissonEncoder(time=time, dt=1), None, "./data/MNIST", train=False, download=True, transform=None)

# Pipeline
pipeline = EnvironmentPipeline(
    network=network,
    environment=train_dataset,
    encoding=PoissonEncoder(time=time, dt=1),
)

# Training

# Create a dataloader for the training set

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Training started...")
for epoch in range(n_epochs):
    for i, (data, label) in enumerate(train_loader):
        if i >= samples_per_epoch:
            break

        # Move everything to the right device!
        data = data.to(device)
        label = label.to(device)
        index = torch.tensor([i]).to(device)
        reward = torch.tensor(0)  # Unused for STDP

        # Let pipeline handle encoding
        pipeline.step(batch=(data, label, index, reward))

        if i % 50 == 0:
            print(f"[Epoch {epoch}] Sample {i}/{samples_per_epoch}")
print("Training complete.")

# Assign neuron labels
print("Assigning labels to neurons...")
assignments, proportions, rates = assign_labels(
    data_loader=pipeline.dataloader,
    network=network,
    layer='Y',
    n_labels=n_classes
)


# Evaluation
def evaluate(dataset, n_samples=100):
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(dataset):
        if i >= n_samples:
            break

        inputs = {'X': PoissonEncoder(time)(images.view(-1))}
        network.run(inputs=inputs, time=time)

        output_spikes = network.monitors['Y'].get('s')
        prediction = all_activity(
            spikes=output_spikes,
            assignments=assignments,
            n_labels=n_classes
        )

        if prediction == labels.item():
            correct += 1
        total += 1
        network.reset_state_variables()

    print(f"Test accuracy: {correct / total:.2%}")


# Add spike monitor
from bindsnet.network.monitors import Monitor

spike_monitor = Monitor(obj=conv_layer, state_vars=['s'], time=time)
network.add_monitor(spike_monitor, name='Y')

# Run evaluation
print("Evaluating on test set...")
evaluate(test_dataset, n_samples=100)

# Optional: Visualize spikes
plot_spikes({'Y': spike_monitor.get('s')})
plt.show()
