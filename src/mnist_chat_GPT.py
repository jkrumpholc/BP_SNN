import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Conv2dConnection
from bindsnet.learning import PostPre
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.pipeline import EnvironmentPipeline

from bindsnet.evaluation import assign_labels
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Device config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
time = 100
n_classes = 10
input_shape = (1, 28, 28)
conv_filters = 32
conv_kernel = 5
conv_stride = 1
n_epochs = 1
samples_per_epoch = 300

# Build network
network = Network(dt=1.0)
network.to(device)

# Layers
input_layer = Input(n=28 * 28, shape=input_shape, traces=True, device="cuda:0")
conv_layer = LIFNodes(n=conv_filters * 24 * 24, shape=(conv_filters, 24, 24), traces=True)

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
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dataset = MNIST(PoissonEncoder(time=time, dt=1), None, "./data/MNIST", train=True, download=True, transform=None)

# Pipeline
pipeline = EnvironmentPipeline(
    network=network,
    environment=train_dataset,
    encoding=PoissonEncoder(time=time, dt=1),
)


def train():
    # Training

    # Create a dataloader for the training set
    time = pipeline.time
    layer = pipeline.network.layers['Y']
    D, W, H = pipeline.network.layers['Y'].shape
    n_neurons = H * W * D
    spike_record = torch.zeros(samples_per_epoch, time, n_neurons)
    labels = torch.zeros(samples_per_epoch, dtype=torch.long)

    # DataLoader
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print("Training started...")
    for epoch in range(n_epochs):
        for i, (data, label) in enumerate(train_loader):
            if i >= samples_per_epoch:
                break

            # Move everything to the right device!
            data = data.squeeze(0)
            data = data.to(device)
            label = label.to(device)
            index = torch.tensor([i]).to(device)
            reward = torch.tensor(0)  # Unused for STDP

            # Let pipeline handle encoding
            pipeline.step(batch=(data, label, index, reward))
            spikes = pipeline.network.layers['Y'].s.detach().cpu().view(time, -1)
            spike_record[i] = spikes
            labels[i] = label

            if i % 50 == 0:
                print(f"[Epoch {epoch}] Sample {i}/{samples_per_epoch}")
    print("Training complete.")

    # Assign neuron labels
    print("Assigning labels to neurons...")
    assignments, proportions, rates = assign_labels(spike_record, labels, 10)
    return assignments, proportions, rates


def all_activity(spikes, assignments, n_labels):
    num_neurons = spikes.size(2) * spikes.size(3) * spikes.size(4)  # Get the number of neurons
    rates = torch.zeros(n_labels).to(spikes.device)  # Store the firing rates for each label
    spikes = spikes.view(spikes.size(0), -1)
    # Ensure indices are within bounds
    for i in range(n_labels):
        indices = assignments == i  # Assign neurons based on their label

        # Ensure indices are within valid range
        indices = indices.nonzero().squeeze()

        # Make sure we don't exceed the number of neurons
        if indices.size(0) > num_neurons:
            raise ValueError(f"Too many indices for label {i}, exceeds the number of neurons.")

        # Calculate the firing rates for each label
        rates[i] = torch.sum(spikes[:, indices], 1).float().mean()

    return rates


# Evaluation
def evaluate(dataset, n_samples=100):
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(test_loader):
        if i >= n_samples:
            break

        # Move images and labels to the correct device (CUDA)
        images = images.to(device)
        labels = labels.to(device)

        # Flatten the image
        image_tensor = images.to(device)  # Flatten to [784]

        # Encode input using PoissonEncoder
        inputs = {'X': PoissonEncoder(time)(image_tensor).to(device)}
        # Run network on GPU
        network.run(inputs=inputs, time=time, device=device)

        output_spikes = network.monitors['Y'].get('s')
        prediction = all_activity(
            spikes=output_spikes,
            assignments=assignments,
            n_labels=n_classes
        )

        pred_label = torch.argmax(prediction).item()
        if pred_label == labels.item():
            correct += 1
        total += 1
        network.reset_state_variables()

    print(f"Test accuracy: {correct / total:.2%}")


# Add spike monitor

spike_monitor = Monitor(obj=conv_layer, state_vars=['s'], time=time)
network.add_monitor(spike_monitor, name='Y')
assignments, proportions, rates = train()
# Run evaluation
print("Evaluating on test set...")
evaluate(test_dataset, n_samples=100)

# Optional: Visualize spikes
# plot_spikes({'Y': spike_monitor.get('s')})
# plt.show()