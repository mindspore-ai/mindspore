import numpy as np
from mindspore import Tensor, Model, context, jit
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


def run_test(netclass, count, dev):
    context.set_context(mode=context.PYNATIVE_MODE, device_target=dev)
    net = netclass()
    model = Model(net)
    for _ in range(count):
        input_np = np.random.randn(2, 3).astype(np.float32)
        input_ms = Tensor(input_np)
        output_np = net.construct(input_np)  # run python
        output_ms = model.predict(input_ms)  # run graph
        np.testing.assert_array_almost_equal(output_np, output_ms.asnumpy(), decimal=3)


class ForLoopWithBreak(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        for i in range(8):
            if i > 5:
                x *= 3
                break
            x = x * 2
        return x


class ForLoopWithContinue(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        for i in range(8):
            if i > 5:
                x *= 3
                continue
            x = x * 2
        return x


class ForLoopWithContBreak(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        for i in range(8):
            if i < 3:
                i *= 2
                continue
            if i > 5:
                x *= 3
                break
            x = x * 2
        return x


class ForNestedLoopWithBreak(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        for _ in range(3):
            for j in range(5):
                if j > 3:
                    x *= 2
                    break
                x = x * 1.5
        return x


class WhileWithBreak(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                x *= 2
                break
            x = x * 1.5
            i += 1
        return x


class WhileWithContinue(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                x *= 2
                i += 1
                continue
            x = x * 1.5
            i += 1
        return x


class WhileForNested(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                for j in range(3):
                    if j > 1:
                        break
                    x *= 2
                i += 1
                continue
            x = x * 1.5
            i += 1
        return x


class PassBranch(Cell):
    @jit(mode="PIJit")
    def construct(self, x):
        i = 0
        while i < 5:
            if i > 3:
                pass
            else:
                x = x * 1.5
            i += 1
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cont_break():
    """
    Feature: Get container with break
    Description: Watching the container with break func graph from abstract.
    Expectation: Output correct.
    """
    count = 20
    dev = 'CPU'
    run_test(ForLoopWithBreak, count, dev)
    run_test(ForLoopWithContinue, count, dev)
    run_test(ForLoopWithContBreak, count, dev)
    run_test(ForNestedLoopWithBreak, count, dev)
    run_test(WhileWithBreak, count, dev)
    run_test(WhileWithContinue, count, dev)
    run_test(WhileForNested, count, dev)
    run_test(PassBranch, count, dev)
