from typing import Iterable, Optional, Sequence, Union

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection


class CustomNetwork(Network):

    def __init__(
        self,
        n_input: int = 784,
        n_hidden: int = 400,
        n_output: int = 10,
        batch_size: int = 1,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        input_shape: Optional[Iterable[int]] = (1, 28, 28),
    ) -> None:
        super().__init__(dt=dt)

        self.n_input = n_input
        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.inh = inh
        self.dt = dt
        self.batch_size = batch_size

        input_layer = Input(
            n=self.n_input, shape=self.input_shape, traces=True, tc_trace=20.0, batch_size=batch_size
        )
        self.add_layer(input_layer, name="X")

        conv_layer = LIFNodes(
            n=12 * 24 * 24,
            traces=True,
            shape=(12, 24, 24)
        )
        self.add_layer(conv_layer, name="Conv")

        conv_conn = Conv2dConnection(
            source=input_layer,
            target=conv_layer,
            kernel_size=5,
            stride=1,
            n_filters=12,
            reduction=reduction,
            update_rule=PostPre,
            nu=nu,
            wmin=wmin, wmax=wmax
        )
        self.add_connection(connection=conv_conn, source='X', target='Conv')

        fc_layer = LIFNodes(
            n=self.n_hidden,
            traces=True
        )

        self.add_layer(fc_layer, name="FC")

        fc_conn = Connection(
            source=conv_layer,
            target=fc_layer,
            update_rule=PostPre,
            wmin=0.0,
            wmax=1.0
        )

        self.add_connection(connection=fc_conn, source='Conv', target='FC')

        output_layer = LIFNodes(
            n=self.n_output,
            traces=True
        )

        self.add_layer(output_layer, name='Y')

        output_conn = Connection(
            source=fc_layer,
            target=output_layer,
            update_rule=PostPre,
            wmin=0.0, wmax=1.0
        )

        self.add_connection(output_conn, source='FC', target='Y')
