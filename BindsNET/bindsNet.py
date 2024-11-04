import torch
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes
from bindsnet.encoding import PoissonEncoder
import matplotlib.pyplot as plt
n_input = 150
n_output = 100
time = 250
dt = 1.0
network = Network(dt=dt)
input_layer = Input(n=n_input)
output_layer = LIFNodes(n=n_output)
network.add_layer(layer=input_layer, name='input')
network.add_layer(layer=output_layer, name='output')
connection = Connection(source=input_layer, target=output_layer)
network.add_connection(connection, source='input', target='output')
input_monitor = Monitor(input_layer, state_vars=['s'])
output_monitor = Monitor(output_layer, state_vars=['s'])
network.add_monitor(input_monitor, name='input')
network.add_monitor(output_monitor, name='output')
encoder = PoissonEncoder(time=time)
input_data = torch.bernoulli(0.1 * torch.ones(time, n_input))
spikes = {'input': input_data}
network.run(inputs=spikes, time=time)
input_spikes = input_monitor.get('s')
output_spikes = output_monitor.get('s')
plt.figure(figsize=(12, 6))
plt.subplot(211)
plot_spikes({'input': input_spikes}, time=(0, time))
plt.title('Input Layer Spikes')
plt.subplot(212)
plot_spikes({'output': output_spikes}, time=(0, time))
plt.title('Output Layer Spikes')
plt.tight_layout()
plt.show()
