import mindspore.ops.operations as op
from mindspore import Tensor, jit, context
from mindspore.nn import Cell
from ...utils import allclose_nparray
from ...utils import tensor_to_numpy


class Less(Cell):
    def __init__(self, left_input, right_input):
        super().__init__()
        self.left_input = left_input
        self.right_input = right_input
        self.less = op.Less()

    def construct(self):
        return self.less(self.left_input, self.right_input)


class LessFactory():
    def __init__(self, left_input, right_input, leftistensor=True, rightistensor=True):
        self.left_input_np = left_input
        self.right_input_np = right_input
        self.leftistensor = leftistensor
        self.rightistensor = rightistensor

    def forward_mindspore_impl(self, net):
        out = net()
        return out.asnumpy()


    def forward_cmp(self):
        if self.leftistensor:
            left_input = Tensor(self.left_input_np)
        else:
            left_input = self.left_input_np
        if self.rightistensor:
            right_input = Tensor(self.right_input_np)
        else:
            right_input = self.right_input_np
        ps_net = Less(left_input, right_input)
        jit(ps_net.construct, mode="PSJit")()
        context.set_context(mode=context.GRAPH_MODE)
        out_psjit = self.forward_mindspore_impl(ps_net)
        pi_net = Less(left_input, right_input)
        jit(pi_net.construct, mode="PIJit")()
        context.set_context(mode=context.PYNATIVE_MODE)
        out_pijit = self.forward_mindspore_impl(pi_net)
        allclose_nparray(out_pijit, out_psjit, 0, 0)


class LessMock(LessFactory):
    def __init__(self, inputs=None):
        input_x = inputs[0]
        input_y = inputs[1]
        leftistensor = False
        rightistensor = False
        if isinstance(inputs[0], Tensor):
            input_x = tensor_to_numpy(input_x)
            leftistensor = True
        if isinstance(inputs[1], Tensor):
            input_y = tensor_to_numpy(input_y)
            rightistensor = True
        LessFactory.__init__(self, input_x, input_y, leftistensor, rightistensor)
