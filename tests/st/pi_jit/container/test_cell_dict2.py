import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, jit
import numpy as np
from collections import OrderedDict
from tests.mark_utils import arg_mark


class TestGetitemMethodNet(nn.Cell):
    def __init__(self):
        super(TestGetitemMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(6, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        return self.cell_dict['conv']


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_getitem_method(mode):
    """
    Feature: CellDict.__getitem__()
    Description: Verify the result of CellDict.__getitem__().
    Expectation: success
    """
    net = TestGetitemMethodNet()
    x = Tensor(np.ones([1, 6, 16, 5]), ms.float32)
    conv2d_op = nn.Conv2d(6, 16, 5, pad_mode='valid')
    expect_output = conv2d_op(x)
    net_op = net()
    output = net_op(x)
    assert np.allclose(output.shape, expect_output.shape)


class TestSetitemMethodNet(nn.Cell):
    def __init__(self):
        super(TestSetitemMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        self.cell_dict['conv'] = nn.Conv2d(6, 16, 5, pad_mode='valid')
        return self.cell_dict['conv']


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_setitem_method(mode):
    """
    Feature: CellDict.__setitem__()
    Description: Verify the result of CellDict.__setitem__().
    Expectation: success
    """
    net = TestSetitemMethodNet()
    x = Tensor(np.ones([1, 6, 16, 5]), ms.float32)
    conv2d_op = nn.Conv2d(6, 16, 5, pad_mode='valid')
    expect_output = conv2d_op(x)
    net_op = net()
    output = net_op(x)
    assert np.allclose(output.shape, expect_output.shape)


class TestSetitemMethodErrCaseNet(nn.Cell):
    def __init__(self):
        super(TestSetitemMethodErrCaseNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self, key, cell):
        self.cell_dict[key] = cell
        return self.cell_dict[key]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_setitem_error_case_method(mode):
    """
    Feature: CellDict.__setitem__()
    Description: Verify the result of CellDict.__setitem__() in error input.
    Expectation: success
    """
    net = TestSetitemMethodErrCaseNet()

    cell = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    key = 1
    with pytest.raises(TypeError):
        net(key, cell)

    cell = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    key = "_scope"
    with pytest.raises(KeyError):
        net(key, cell)

    cell = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    key = ".conv1d"
    with pytest.raises(KeyError):
        net(key, cell)

    cell = nn.Conv1d(120, 240, 4, has_bias=False, weight_init='normal')
    key = ""
    with pytest.raises(KeyError):
        net(key, cell)

    cell = None
    key = "conv1d"
    with pytest.raises(TypeError):
        net(key, cell)

    cell = 1
    key = "conv1d"
    with pytest.raises(TypeError):
        net(key, cell)

class TestDelitemMethodNet(nn.Cell):
    def __init__(self):
        super(TestDelitemMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    #TODO: fix pijit assert error
    def construct(self, key1, key2):
        del self.cell_dict[key1]
        del self.cell_dict[key2]
        return len(self.cell_dict)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_delitem_method(mode):
    """
    Feature: CellDict.__delitem__()
    Description: Verify the result of CellDict.__delitem__().
    Expectation: success
    """
    net = TestDelitemMethodNet()
    expect_output = 1
    output = net('conv', 'relu')
    assert np.allclose(output, expect_output)


class TestContainsMethodNet(nn.Cell):
    def __init__(self):
        super(TestContainsMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self, key1, key2):
        ret1 = key1 in self.cell_dict
        ret2 = key2 in self.cell_dict
        return ret1, ret2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_contains_method(mode):
    """
    Feature: CellDict.__contains__()
    Description: Verify the result of CellDict.__contains__().
    Expectation: success
    """
    net = TestContainsMethodNet()
    expect_output1 = True
    expect_output2 = False
    output1, output2 = net('conv', 'relu1')
    assert expect_output1 == output1
    assert expect_output2 == output2


class TestClearMethodNet(nn.Cell):
    def __init__(self):
        super(TestClearMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    #@jit(mode="PIJit")
    def construct(self):
        self.cell_dict.clear()
        return len(self.cell_dict)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_clear_method(mode):
    """
    Feature: CellDict.clear()
    Description: Verify the result of CellDict.clear().
    Expectation: success
    """
    net = TestClearMethodNet()
    expect_output = 0
    output = net()
    assert np.allclose(expect_output, output)


class TestPopMethodNet(nn.Cell):
    def __init__(self):
        super(TestPopMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    #@jit(mode="PIJit")
    def construct(self, key):
        op = self.cell_dict.pop(key)
        cell_dict_len = len(self.cell_dict)
        return op, cell_dict_len


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_pop_method(mode):
    """
    Feature: CellDict.pop()
    Description: Verify the result of CellDict.pop().
    Expectation: success
    """
    net = TestPopMethodNet()
    conv_op = nn.Conv2d(10, 16, 5, pad_mode='valid')
    x = Tensor(np.ones([1, 10, 6, 5]), ms.float32)
    expect_output = conv_op(x)
    expect_len = 2
    op, cell_dict_len = net('conv')
    output = op(x)
    assert np.allclose(expect_output.shape, output.shape)
    assert np.allclose(expect_len, cell_dict_len)


class TestKeysMethodNet(nn.Cell):
    def __init__(self):
        super(TestKeysMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        return self.cell_dict.keys()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_keys_method(mode):
    """
    Feature: CellDict.keys()
    Description: Verify the result of CellDict.keys().
    Expectation: success
    """
    net = TestKeysMethodNet()
    expect_keys = ['conv', 'relu', 'max_pool2d']
    cell_dict_keys = net()
    for key, expect_key in zip(cell_dict_keys, expect_keys):
        assert key == expect_key


class TestValuesMethodNet(nn.Cell):
    def __init__(self):
        super(TestValuesMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        return self.cell_dict.values()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_values_method(mode):
    """
    Feature: CellDict.values()
    Description: Verify the result of CellDict.values().
    Expectation: success
    """
    net = TestValuesMethodNet()
    conv2d_op = nn.Conv2d(10, 16, 5, pad_mode='valid')
    relu_op = nn.ReLU()
    maxpool2d_op = nn.MaxPool2d(kernel_size=4, stride=4)
    x = Tensor(np.ones([1, 10, 16, 10]), ms.float32)
    expect_x = conv2d_op(x)
    expect_x = relu_op(expect_x)
    expect_x = maxpool2d_op(expect_x)

    cell_dict_values = net()
    for cell in cell_dict_values:
        x = cell(x)

    assert np.allclose(x.shape, expect_x.shape)


class TestItemsMethodNet(nn.Cell):
    def __init__(self):
        super(TestItemsMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='valid')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        return self.cell_dict.items()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_items_method(mode):
    """
    Feature: CellDict.items()
    Description: Verify the result of CellDict.items().
    Expectation: success
    """
    net = TestItemsMethodNet()
    expect_keys = ['conv', 'relu', 'max_pool2d']
    cell_dict_items = net()
    for item, expect_key in zip(cell_dict_items, expect_keys):
        assert item[0] == expect_key

    conv2d_op = nn.Conv2d(10, 16, 5, pad_mode='valid')
    relu_op = nn.ReLU()
    maxpool2d_op = nn.MaxPool2d(kernel_size=4, stride=4)
    x = Tensor(np.ones([1, 10, 16, 10]), ms.float32)
    expect_x = conv2d_op(x)
    expect_x = relu_op(expect_x)
    expect_x = maxpool2d_op(expect_x)
    for item in cell_dict_items:
        x = item[1](x)
    assert np.allclose(x.shape, expect_x.shape)


class TestUpdateMethodNet(nn.Cell):
    def __init__(self):
        super(TestUpdateMethodNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='same')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )

    @jit(mode="PIJit")
    def construct(self):
        x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
        y = Tensor(np.ones([1, 10, 6, 10]), ms.float32)

        # 用包含键值对的列表更新CellDict
        self.cell_dict.clear()
        cell_list = [['dense1', nn.Dense(3, 4)], ['dense2', nn.Dense(4, 6)], ['dense3', nn.Dense(6, 8)]]
        self.cell_dict.update(cell_list)
        output1 = x
        for cell in self.cell_dict.values():
            output1 = cell(output1)

        # 用OrderDict更新CellDict
        self.cell_dict.clear()
        cell_order_dict = OrderedDict([('conv', nn.Conv2d(10, 6, 5, pad_mode='same')),
                                       ('relu', nn.ReLU()),
                                       ('max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4))]
                                      )
        self.cell_dict.update(cell_order_dict)
        output2 = y
        for cell in self.cell_dict.values():
            output2 = cell(output2)

        # 用CellDict更新CellDict
        self.cell_dict.clear()
        cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 6, 5, pad_mode='same')],
                                 ['relu', nn.ReLU()],
                                 ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                )
        self.cell_dict.update(cell_dict)
        output3 = y
        for cell in self.cell_dict.values():
            output3 = cell(output3)

        return output1, output2, output3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_update_method(mode):
    """
    Feature: CellDict.update()
    Description: Verify the result of CellDict.update().
    Expectation: success
    """
    net = TestUpdateMethodNet()
    x = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
    y = Tensor(np.ones([1, 10, 6, 10]), ms.float32)

    dense_op1 = nn.Dense(3, 4)
    dense_op2 = nn.Dense(4, 6)
    dense_op3 = nn.Dense(6, 8)
    expect_dense_output = x
    expect_dense_output = dense_op1(expect_dense_output)
    expect_dense_output = dense_op2(expect_dense_output)
    expect_dense_output = dense_op3(expect_dense_output)

    conv2d_op = nn.Conv2d(10, 6, 5, pad_mode='same')
    relu_op = nn.ReLU()
    maxpool2d_op = nn.MaxPool2d(kernel_size=4, stride=4)
    expect_output = y
    expect_output = conv2d_op(expect_output)
    expect_output = relu_op(expect_output)
    expect_output = maxpool2d_op(expect_output)

    output1, output2, output3 = net()
    assert np.allclose(expect_dense_output.shape, output1.shape)
    assert np.allclose(expect_output.shape, output2.shape)
    assert np.allclose(expect_output.shape, output3.shape)


class TestUpdateMethodEmbeddedNet(nn.Cell):
    def __init__(self):
        super(TestUpdateMethodEmbeddedNet, self).__init__()
        self.cell_dict = nn.CellDict([['conv', nn.Conv2d(10, 16, 5, pad_mode='same')],
                                      ['relu', nn.ReLU()],
                                      ['max_pool2d', nn.MaxPool2d(kernel_size=4, stride=4)]]
                                     )
    @jit(mode="PIJit")
    def construct(self, object_list):
        self.cell_dict.update(object_list)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_celldict_update_method_embedded_case(mode):
    """
    Feature: CellDict.update()
    Description: Verify the result of CellDict.update() in embedded_case.
    Expectation: success
    """
    net = TestUpdateMethodEmbeddedNet()
    cell_dict = nn.CellDict({'conv': nn.Conv2d(1, 1, 3), 'Dense': nn.Dense(2, 2)})
    cell_list = nn.CellList([nn.Dense(2, 2)])
    conv = nn.Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones")
    relu = nn.ReLU()
    seq_cell = nn.SequentialCell([conv, relu])

    celldict_embedded_list = [['cell_dict', cell_dict]]
    celllist_embedded_list = [['cell_list', cell_list]]
    seqcell_embedded_list = [['seq_cell', seq_cell]]

    with pytest.raises(TypeError):
        net(celldict_embedded_list)

    with pytest.raises(TypeError):
        net(celllist_embedded_list)

    with pytest.raises(TypeError):
        net(seqcell_embedded_list)

class DupParaNameNet1(nn.Cell):
    def __init__(self):
        super(DupParaNameNet1, self).__init__()
        self.cell_dict1 = nn.CellDict({'conv2d': nn.Conv2d(20, 20, 5),
                                       'pool2d': nn.MaxPool2d(7)}
                                      )
        self.cell_dict2 = nn.CellDict({'conv2d': nn.Conv2d(20, 20, 5),
                                       'pool2d': nn.MaxPool2d(7)}
                                      )

    @jit(mode="PIJit")
    def construct(self, x1, x2):
        a = self.cell_dict1['conv2d'](x1)
        b = self.cell_dict2['conv2d'](x2)
        return a + b


class DupParaNameNet2(nn.Cell):
    def __init__(self):
        super(DupParaNameNet2, self).__init__()
        self.cell_dict1 = nn.CellDict({'dense': nn.Dense(3, 4)})
        self.cell_dict2 = nn.CellDict({'dense': nn.Dense(3, 4)})

    @jit(mode="PIJit")
    def construct(self, x1, x2):
        a = self.cell_dict1['dense'](x1)
        b = self.cell_dict2['dense'](x2)
        return a + b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_duplicate_para_name_case(mode):
    """
    Feature: Verify the same parameter names of two CellDicts within the same net can be distinguished.
    Description: Within a net, constructing two CellDicts which are same.
    Expectation: success
    """
    net = DupParaNameNet1()
    x1 = Tensor(np.ones([1, 20, 20, 10]), ms.float32)
    x2 = Tensor(np.ones([1, 20, 20, 1]), ms.float32)
    output = net(x1, x2)
    expect_output_shape = (1, 20, 20, 10)
    assert np.allclose(output.shape, expect_output_shape)

    net = DupParaNameNet2()
    x1 = Tensor(np.array([[180, 234, 154], [244, 48, 247]]), ms.float32)
    x2 = Tensor(np.array([[110, 134, 150], [224, 148, 347]]), ms.float32)
    output = net(x1, x2)
    expect_output_shape = (2, 4)
    assert np.allclose(output.shape, expect_output_shape)
