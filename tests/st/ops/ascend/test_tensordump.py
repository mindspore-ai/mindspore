import pytest
import numpy as np
import time
import tempfile
from pathlib import Path

from mindspore import Tensor, nn, ops
import mindspore as ms


def find_npy_files(folder_path):
    folder_path = Path(folder_path)
    result = {}
    for file in folder_path.glob('*.npy'):
        file_name = file.stem
        file_name_without_id = file_name.split('_')[-1]
        result[file_name_without_id] = str(file.absolute())
    return result


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize(
    "dtype",
    [
        ms.float16,
        ms.float32,
        ms.float64,
        ms.int8,
        ms.uint8,
        ms.int16,
        ms.uint16,
        ms.int32,
        ms.uint32,
        ms.int64,
        ms.uint64,
    ],
)
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(dtype, mode):
    """
    Feature: Check TensorDump ops
    Description: Check TensorDump ops
    Expectation: pass
    """

    class Net(nn.Cell):
        def __init__(self, path):
            super(Net, self).__init__()
            self.dump = ops.TensorDump()
            self.path_x1 = str(path / "x1")
            self.path_x2 = str(path / "x2")
            self.path_out = str(path / "out")

        def construct(self, x1, x2):
            self.dump(self.path_x1, x1)
            self.dump(self.path_x2, x2)
            x1 = ops.cast(x1, ms.float32)
            x2 = ops.cast(x2, ms.float32)
            out = x1 + x2
            self.dump(self.path_out, out)
            return out

    ms.set_context(device_target="Ascend", mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x1 = ops.randint(0, 10, (3, 5)).astype(dtype)
    x2 = ops.randint(0, 10, (3, 5)).astype(dtype)
    net = Net(path)
    output = net(x1, x2)
    out = output.asnumpy()
    time.sleep(0.1)
    name2file = find_npy_files(path)

    assert np.allclose(x1.asnumpy(), np.load(name2file["x1"]))
    assert np.allclose(x2.asnumpy(), np.load(name2file["x2"]))
    assert np.allclose(out, np.load(name2file["out"]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net_bool(mode):
    """
    Feature: Check TensorDump ops
    Description: Check TensorDump ops
    Expectation: pass
    """

    class Net(nn.Cell):
        def __init__(self, path):
            super(Net, self).__init__()
            self.dump = ops.TensorDump()
            self.path_x1 = str(path / "x1")
            self.path_x2 = str(path / "x2")
            self.path_out = str(path / "out")

        def construct(self, x1, x2):
            self.dump(self.path_x1, x1)
            self.dump(self.path_x2, x2)
            out = ops.logical_and(x1, x2)
            self.dump(self.path_out, out)
            return out

    ms.set_context(device_target="Ascend", mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x1 = Tensor(np.array([True, False, True]), ms.bool_)
    x2 = Tensor(np.array([True, True, False]), ms.bool_)
    net = Net(path)
    output = net(x1, x2)
    out = output.asnumpy()
    time.sleep(0.1)
    name2file = find_npy_files(path)
    assert np.allclose(x1.asnumpy(), np.load(name2file["x1"]))
    assert np.allclose(x2.asnumpy(), np.load(name2file["x2"]))
    assert np.allclose(out, np.load(name2file["out"]))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensordump_when_jit(mode):
    """
    Feature: Check TensorDump ops
    Description: Check TensorDump ops when pynative jit
    Expectation: pass
    """
    @ms.jit
    def dump_tensor(x, path):
        ops.TensorDump()(path + "/input", x)
        x1 = x + 1.
        ops.TensorDump()(path + "/add", x1)
        x2 = x1 / 2.
        ops.TensorDump()(path + "/div", x2)
        x3 = x2 * 5
        ops.TensorDump()(path + "/mul", x3)
        return x, x1, x2, x3

    ms.set_context(mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32)
    input_x = Tensor(x)
    x, x1, x2, x3 = dump_tensor(input_x, str(path))
    time.sleep(0.1)
    name2file = find_npy_files(path)
    assert np.allclose(x.asnumpy(), np.load(name2file["input"]))
    assert np.allclose(x1.asnumpy(), np.load(name2file["add"]))
    assert np.allclose(x2.asnumpy(), np.load(name2file["div"]))
    assert np.allclose(x3.asnumpy(), np.load(name2file["mul"]))
