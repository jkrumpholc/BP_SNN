from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch.nn.modules.utils import _pair
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection

from bindsnet.network.monitors import Monitor


class CAModel(Network):
    def __init__(
            self,
            n_inpt: int,
            input_shape: List[int],
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]],
            n_filters: int,
            inh: float = 25.0,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            wmin: float = 0.0,
            wmax: float = 1.0,
            norm: Optional[float] = 0.2,
            exc_thresh: float = -52.0,
            n_channels=16,
            fire_rate=0.5,
            device=None,
    ):
        super().__init__()

        self.fire_rate = fire_rate
        self.n_channels = n_channels
        self.device = device or torch.device("cpu")

        # Perceive step
        sobel_filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0

        sobel_filter_x = sobel_filter_ / scalar
        sobel_filter_y = sobel_filter_.t() / scalar
        identity_filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0], ], dtype=torch.float32, )
        filters = torch.stack([identity_filter, sobel_filter_x, sobel_filter_y])  # (3, 3, 3)
        filters = filters.repeat((n_channels, 1, 1))  # (3 * n_channels, 3, 3)
        self.filters = filters[:, None, ...].to(self.device)  # (3 * n_channels, 1, 3, 3)

        super().__init__(dt=dt)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.n_inpt = n_inpt
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters
        self.inh = inh
        self.dt = dt
        self.theta_plus = theta_plus
        self.tc_theta_decay = tc_theta_decay
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm

        if kernel_size == input_shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((input_shape[1] - kernel_size[0]) / stride[0]) + 1,
                int((input_shape[2] - kernel_size[1]) / stride[1]) + 1,
            )

        input_layer = Input(n=self.n_inpt, shape=input_shape, traces=True, tc_trace=20.0)

        middle_layer = DiehlAndCookNodes(
            n=self.n_filters * conv_size[0] * conv_size[1],
            shape=[1, 32, 32],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        output_layer = LIFNodes(
            n=self.n_filters * conv_size[0] * conv_size[1] / stride[0] / stride[1],
            shape=[1, 16, 16],
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )

        conv_conn_1 = Conv2dConnection(
            input_layer,
            middle_layer,
            kernel_size=kernel_size,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        conv_conn_2 = Conv2dConnection(
            middle_layer,
            output_layer,
            kernel_size=kernel_size*4,
            stride=stride,
            n_filters=n_filters,
            nu=nu,
            reduction=reduction,
            update_rule=PostPre,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
            input_shape=input_shape,
        )

        w = torch.zeros(n_filters, *conv_size, n_filters, *conv_size)
        for fltr1 in range(n_filters):
            for fltr2 in range(n_filters):
                if fltr1 != fltr2:
                    for i in range(conv_size[0]):
                        for j in range(conv_size[1]):
                            w[fltr1, i, j, fltr2, i, j] = -inh

        w = w.view(
            n_filters * conv_size[0] * conv_size[1],
            n_filters * conv_size[0] * conv_size[1],
        )
        recurrent_conn = Connection(output_layer, output_layer, w=w)

        self.add_layer(input_layer, name="X")
        self.add_layer(middle_layer, name="Y")
        self.add_layer(output_layer, name="Z")
        self.add_connection(conv_conn_1, source="X", target="Y")
        self.add_connection(conv_conn_2, source="Y", target="Z")
        self.add_connection(recurrent_conn, source="Z", target="Z")

        monitor_1 = Monitor(self.layers["Y"], ["v"], time=int(100 / dt), device=device)
        monitor_2 = Monitor(self.layers["Z"], ["v"], time=int(100 / dt), device=device)
        self.add_monitor(monitor_1, "middle_voltage")
        self.add_monitor(monitor_2, "output_voltage")
