import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.communication as comm
import mindspore.dataset as ds


class GPTONetDataset:
    def __init__(self, tensors):
        self.data = []
        self.label = []

        for i, tensor in enumerate(tensors):
            self.data.append(tensor)
            self.label.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label


class GPTONet(nn.Cell):
    def __init__(self):
        super(GPTONet, self).__init__()
        self.add_branch_1 = ops.Add()
        self.sub_branch_1 = ops.Sub()
        self.sub_branch_2 = ops.Sub()
        self.allreduce_branch_2 = ops.AllReduce()
        self.mul_output = ops.Mul()

    def overlap_net(self, i):
        o1 = self.add_branch_1(i, i)
        o1 = self.sub_branch_1(i, o1)
        o2 = self.sub_branch_2(i, i)
        o2 = self.allreduce_branch_2(o2)
        output = self.mul_output(o1, o2)
        return output

    def construct(self, i, t1=None):
        output = self.overlap_net(i)
        return output


def test_gpto_net():
    """
    Feature: this function test the GPTO module in KBK
    Description: the input is a net with comp and comm operators, gpto will overlap them
    Expectation: the test should pass without any error and exception
    """
    ms.set_context(
        jit_config={"jit_level": "O0"},
        memory_optimize_level="O1",
        device_target="Ascend",
        mode=ms.GRAPH_MODE,
        exec_order="gpto",
    )
    comm.init()
    network = GPTONet()
    x = ms.Tensor(np.random.randn(512, 512), dtype=ms.float32)
    gpto_net_dataset = GPTONetDataset([x, x])
    dataset = ds.GeneratorDataset(gpto_net_dataset, column_names=["data", "label"])
    model = ms.Model(network)
    model.train(1, dataset)


test_gpto_net()
