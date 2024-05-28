import numpy as np
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import Tensor
from ..share.utils import allclose_nparray


class DynamicShapeInplaceAdd(Cell):
    def __init__(self, indices):
        super().__init__()
        self.inplaceadd = P.InplaceAdd(indices)
        self.reducesum = P.ReduceSum(keep_dims=False)
        self.relu = P.ReLU()

    def construct(self, x, v, select_indices):
        unique_indices = self.relu(select_indices)
        x = self.reducesum(x, unique_indices)
        v = self.reducesum(v, unique_indices)
        out = self.inplaceadd(x, v)
        return out


class InplaceAddDynamicShapeMock():
    def __init__(self, inputs=None, attributes=None):
        self.indices = attributes.get('indices')
        self.input_x = inputs[0]
        self.input_x_np = inputs[0].asnumpy()
        self.input_v = inputs[1]
        self.input_v_np = inputs[1].asnumpy()
        self.select_indices = inputs[2]
        self.select_indices_np = inputs[2].asnumpy()

    def forward_mindspore_dynamic_shape_impl(self, net):
        input_x_dyn = Tensor(shape=[None for _ in self.input_x.shape], dtype=self.input_x.dtype)
        input_v_dyn = Tensor(shape=[None for _ in self.input_v.shape], dtype=self.input_x.dtype)
        select_indices_dyn = Tensor(shape=[None], dtype=self.select_indices.dtype)
        net.set_inputs(input_x_dyn, input_v_dyn, select_indices_dyn)
        out = net(self.input_x, self.input_v, self.select_indices)
        return out.asnumpy()

    def forward_dynamic_shape_cmp(self):
        ps_net = DynamicShapeInplaceAdd(self.indices)
        jit(ps_net.construct, mode="PSJit")(self.input_x, self.input_v, self.select_indices)
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_dynamic_shape_impl(ps_net)

        pi_net = DynamicShapeInplaceAdd(self.indices)
        jit(pi_net.construct, mode="PIJit")(self.input_x, self.input_v, self.select_indices)
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_dynamic_shape_impl(pi_net)

        allclose_nparray(out_pijit, out_psjit, self.loss, self.loss)


def generate_random_shape(dimension, num_ones, x_d0, v_d0):
    assert dimension > num_ones
    shape = np.random.randint(1, 17, size=dimension)
    i = 0
    list_1 = []
    while i < dimension - 1:
        i = i + 1
        list_1.append(i)
    indices = np.random.choice(list_1, size=num_ones, replace=False)
    shape[indices] = 1
    v_shape = list(shape)
    shape[0] = x_d0
    v_shape[0] = v_d0
    return shape, v_shape, indices
