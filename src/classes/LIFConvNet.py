from norse.torch.models.conv import ConvNet4
from norse.torch.module.encode import ConstantCurrentLIFEncoder
import torch
import torch.utils.data


class LIFConvNet(torch.nn.Module):
    def __init__(
        self,
        input_features,
        seq_length,
        input_scale,
        model="super",
    ):
        super(LIFConvNet, self).__init__()
        self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
        self.input_features = input_features
        self.rsnn = ConvNet4(method=model)
        self.seq_length = seq_length
        self.input_scale = input_scale
        self.voltages = 0

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.constant_current_encoder(
            x.view(-1, self.input_features) * self.input_scale
        )
        x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
        self.voltages = self.rsnn(x)
        m, _ = torch.max(self.voltages, 0)
        log_p_y = torch.nn.functional.log_softmax(m, dim=1)
        return log_p_y
