from tests.mark_utils import arg_mark
import pytest
import numpy as np
import time
import tempfile
from pathlib import Path

from mindspore import Tensor, nn, ops
import mindspore as ms
from mindspore.ops.primitive import _run_op
from mindspore import hal
from mindspore._c_expression import _tensordump_set_step


def find_npy_files(folder_path):
    folder_path = Path(folder_path)
    result = {}
    for file in folder_path.glob('*.npy'):
        file_name = file.stem
        file_name_without_id = file_name.split('_')[0]
        result[file_name_without_id] = str(file.absolute())
    return result


def validate_files(root_dir, rank_dir, expected_steps, expected_values):
    """
    Validate the directory structure and file contents.

    Args:
        root_dir (str or Path): The root directory, e.g., "dump_data".
        rank_dir (str or Path): The rank directory, e.g., "rank0".
        expected_steps (list): List of expected `step` subdirectories, e.g., ["step0", "step2"].
        expected_values (dict): Expected file contents in the format:
            {
                "step0/add_bfloat16_0.npy": np.array([...]),
                "step0/mul_bfloat16_1.npy": np.array([...]),
                "step2/add_bfloat16_2.npy": np.array([...]),
                "step2/mul_bfloat16_3.npy": np.array([...]),
            }

    Returns:
        None. All validations are performed using assertions.
    """
    root_path = Path(root_dir)
    rank_path = root_path / rank_dir

    assert root_path.exists(), f"Root directory does not exist: {root_path}"
    assert rank_path.exists(), f"Rank directory does not exist: {rank_path}"

    actual_steps = sorted([p.name for p in rank_path.iterdir() if p.is_dir()])
    expected_steps = sorted(expected_steps)

    # Assert the `step` directories match the expected list
    assert actual_steps == expected_steps, (
        f"Step directories do not match. Actual: {actual_steps}, Expected: {expected_steps}"
    )

    # Validate each file in the expected steps
    for step_dir in expected_steps:
        for relative_path, expected_array in expected_values.items():
            if not relative_path.startswith(step_dir):
                continue
            file_path = rank_path / relative_path
            assert file_path.exists(), f"File does not exist: {file_path}"
            actual_array = np.load(file_path)

            assert np.allclose(actual_array, expected_array), (
                f"File content mismatch: {file_path}. "
                f"Actual: {actual_array}, Expected: {expected_array}"
            )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1.00)
@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
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
            self.path_x = str(path / "x")

        def construct(self, x):
            self.dump(self.path_x, x)
            return x

    ms.set_context(device_target="Ascend", mode=mode)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    np_x = np.random.rand(3, 5)
    x = ms.Tensor(np_x.astype(dtype))
    net = Net(path)
    output = net(x)
    out = output.asnumpy()
    time.sleep(1)
    name2file = find_npy_files(path)

    assert np.allclose(out, np.load(name2file["x"]))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
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
    time.sleep(1)
    name2file = find_npy_files(path)
    assert np.allclose(x1.asnumpy(), np.load(name2file["x1"]))
    assert np.allclose(x2.asnumpy(), np.load(name2file["x2"]))
    assert np.allclose(out, np.load(name2file["out"]))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.25)
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
    x.asnumpy()
    time.sleep(1)
    name2file = find_npy_files(path)
    assert np.allclose(x.asnumpy(), np.load(name2file["input"]))
    assert np.allclose(x1.asnumpy(), np.load(name2file["add"]))
    assert np.allclose(x2.asnumpy(), np.load(name2file["div"]))
    assert np.allclose(x3.asnumpy(), np.load(name2file["mul"]))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.95)
def test_bfloat16():
    """
    Feature: Check TensorDump ops
    Description: Check TensorDump ops when the data type is bfloat16.
    Expectation: pass
    """

    class Net(nn.Cell):
        def __init__(self, path):
            super(Net, self).__init__()
            self.dump = ops.TensorDump()
            self.path = str(path / "input")

        def construct(self, x):
            self.dump(self.path, x)
            x += 1
            return x

    ms.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump")
    path = Path(temp_dir.name)
    x_np = np.ones((3, 5), dtype=np.float32)
    x = ms.Tensor(x_np, ms.bfloat16)
    net = Net(path)
    net(x)
    time.sleep(1)
    filename = "input_bfloat16_0.npy"
    input_x = np.load(str(path) + "/" + filename)
    assert np.allclose(input_x, x_np)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_tensordump_when_graph_pynative_hybrid_rank_step():
    """
    Feature: Check TensorDump ops
    Description: This test validates the behavior of the TensorDump operator in a hybrid execution scenario
                 combining graph and PyNative modes. It ensures the correct dumping of tensors for specified steps
                 and ranks during execution.
    Expectation: The directory structure and dumped tensor values match the expectations. The test passes if:
                 1. The correct step directories (e.g., "step0", "step2") are created.
                 2. The expected tensor dump files exist (e.g., "add_float16_0.npy", "mul_float16_1.npy").
                 3. The dumped tensor values are accurate.
    """
    class Net(nn.Cell):
        def __init__(self, path):
            super(Net, self).__init__()
            self.dump = ops.TensorDump()
            self.add_path = str(path / "{rank}" / "{step}" / "add")
            self.mul_path = str(path / "{rank}" / "{step}" / "mul")

        @ms.jit
        def compute(self, x):
            x = x * 2
            self.dump(self.mul_path, x)
            return x

        def construct(self, x):
            x = x + 1
            self.dump(self.add_path, x)
            x = self.compute(x)
            return x

    def step():
        hal.synchronize()
        temp_tensor = ms.Tensor([1], dtype=ms.float32)
        step_flag = "<tensordump-update-step>"
        _run_op(ops.TensorDump(), "TensorDump", (step_flag, temp_tensor))
        ops.tensordump(step_flag, temp_tensor)
        hal.synchronize()
        time.sleep(2)

    ms.set_context(device_target="Ascend")
    temp_dir = tempfile.TemporaryDirectory(suffix="TensorDump_step_rank")
    path = Path(temp_dir.name)
    np_data = np.array([1, 2, 3, 4], dtype=np.float16)
    target_add = np_data + 1
    target_mul = target_add * 2
    x = ms.Tensor(np_data)
    net = Net(path)
    _tensordump_set_step([0, 2])
    for _ in range(4):
        net(x)
        step()

    validate_files(temp_dir.name, "rank0", ["step0", "step2"], {
        "step0/add_float16_0.npy": target_add,
        "step0/mul_float16_1.npy": target_mul,
        "step2/add_float16_2.npy": target_add,
        "step2/mul_float16_3.npy": target_mul,
    })
