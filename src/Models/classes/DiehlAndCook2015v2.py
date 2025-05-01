from typing import Iterable, Optional, Sequence, Union

import torch

from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Connection

class DiehlAndCook2015v2(Network):
    # language=rst
    """
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        batch_size: int = 1,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt
        self.batch_size = batch_size

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0, batch_size=batch_size
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
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
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        self.add_connection(recurrent_connection, source="Y", target="Y")