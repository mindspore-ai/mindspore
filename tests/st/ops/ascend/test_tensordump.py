import pytest
import numpy as np
import time
import tempfile
from pathlib import Path

from mindspore import Tensor, nn, ops
import mindspore as ms


def find_npy_file(folder_path):
    folder_path = Path(folder_path)
    npy_files = list(folder_path.glob('*.npy'))
    res = None

    if not npy_files:
        raise FileNotFoundError("No .npy file found in the folder.")
    if len(npy_files) == 1:
        res = str(npy_files[0])
    if len(npy_files) > 1:
        raise ValueError(
            "Multiple .npy files found in the folder. There should be only one.")
    return res


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
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
            self.path = str(path / "out")

        def construct(self, x1, x2):
            out = x1 + x2
            self.dump(self.path, out)
            return out

    ms.set_context(device_target="Ascend", mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x1 = ops.randint(0, 10, (3, 5)).astype(dtype)
    x2 = ops.randint(0, 10, (3, 5)).astype(dtype)
    net = Net(path)
    output = net(Tensor(x1), Tensor(x2))
    out = output.asnumpy()
    time.sleep(0.1)
    assert np.allclose(out, np.load(find_npy_file(path)))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
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
            self.path = str(path / "out")

        def construct(self, x1, x2):
            out = ops.logical_and(x1, x2)
            self.dump(self.path, out)
            return out

    ms.set_context(device_target="Ascend", mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x1 = Tensor(np.array([True, False, True]), ms.bool_)
    x2 = Tensor(np.array([True, True, False]), ms.bool_)
    net = Net(path)
    output = net(Tensor(x1), Tensor(x2))
    out = output.asnumpy()
    time.sleep(0.1)
    assert np.allclose(out, np.load(find_npy_file(path)))
