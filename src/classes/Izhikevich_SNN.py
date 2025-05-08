import torch
import torch.nn as nn
from norse.torch import poisson_encode
from norse.torch.module.izhikevich import Izhikevich, IzhikevichSpikingBehavior
from norse.torch.functional.izhikevich import IzhikevichParameters, IzhikevichState


class IzhikevichSNN(nn.Module):
    def __init__(self, in_features=784, hidden_features=256, out_features=10, time_steps=100):
        super().__init__()
        self.time_steps = time_steps
        self.state = IzhikevichState(v=torch.full((hidden_features,), -65.0, device=torch.device("cuda")),
                                     u=torch.full((hidden_features,), -13.0, device=torch.device("cuda")))
        self.params = IzhikevichParameters(a=0.02, b=0.2, c=-50, d=2)
        self.behavior = IzhikevichSpikingBehavior(p=self.params, s=self.state)
        self.fc = nn.Linear(in_features, hidden_features)
        self.out = nn.Linear(hidden_features, out_features)
        self.layer1 = Izhikevich(spiking_method=self.behavior)

    def forward(self, inp):
        self.behavior = IzhikevichSpikingBehavior(p=self.params, s=self.state)
        batch_size = inp.shape[0]
        spk_out = torch.zeros(self.time_steps, batch_size, 10, device=inp.device)

        x_encoded = poisson_encode(inp, self.time_steps)

        for t in range(self.time_steps):
            x = self.fc(x_encoded[t])
            y, self.state = self.layer1(x)
            z = self.out(y)
            spk_out[t] = z

        return spk_out.sum(dim=0)  # sum over time
