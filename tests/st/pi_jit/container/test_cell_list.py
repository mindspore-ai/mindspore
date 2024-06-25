import pytest
import mindspore as ms
from mindspore import nn, jit
import numpy as np
from tests.mark_utils import arg_mark


class TestCellListInsertNet(nn.Cell):
    def __init__(self):
        super(TestCellListInsertNet, self).__init__()
        self.cell_list = nn.CellList()
        self.cell_list.insert(0, nn.Cell())
        self.cell_list.insert(1, nn.Dense(1, 2))

    @jit(mode="PIJit")
    def construct(self):
        return len(self.cell_list)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celllist_insert_method_boundary_cond(mode):
    """
    Feature: CellList.insert()
    Description: Verify the result of CellDict.insert(index, cell) in boundary conditions.
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = TestCellListInsertNet()
    expect_output = 2
    output = net()
    assert np.allclose(output, expect_output)
    x = nn.Dense(1, 2)
    assert type(x) is type(net.cell_list[1])


class EmbeddedCellDictNet(nn.Cell):
    def __init__(self):
        super(EmbeddedCellDictNet, self).__init__()
        self.cell_dict = nn.CellDict({'conv': nn.Conv2d(3, 2, 2), "relu": nn.ReLU()})
        self.cell_list = nn.CellList([self.cell_dict])

    @jit(mode="PIJit")
    def construct(self, x):
        for cell_dict in self.cell_list:
            for net in cell_dict.values():
                x = net(x)
        return x

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celllist_embed_celldict_case(mode):
    """
    Feature: CellList.extend()
    Description: Verify the result of initializing CellList by CellDict
    Expectation: success
    """
    with pytest.raises(TypeError):
        EmbeddedCellDictNet()
