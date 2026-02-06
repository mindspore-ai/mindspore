import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore.communication import init
from mindspore.communication.comm_func import gather_into_tensor
from mindspore.communication.management import get_rank

# 'gather_into_tensor' function only supports KernelByKernel mode by now.
np.random.seed(1)
ms.set_context(jit_level='O0')
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
init()

class GatherIntoTensorFuncNet(nn.Cell):
    def construct(self, tensor, dst):
        out = gather_into_tensor(tensor, dst)
        return out

def test_hccl_gather_into_tensor_func_in_cell_2p():
    """
    Feature: test 'gather_into_tensor' communication function in cell.
    Description: test 'gather_into_tensor' communication function in cell.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(64).reshape([8, 8]).astype(np.float32))
    net = GatherIntoTensorFuncNet()
    out = net(data, 0)
    print(out)
    if rank == 0:
        d0 = np.arange(64).reshape([8, 8]).astype(np.float32)
        gt_rank0 = np.vstack([d0, d0])
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.array([0])
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst

def test_hccl_gather_into_tensor_func_2p():
    """
    Feature: test 'gather_into_tensor' communication function.
    Description: test 'gather_into_tensor' communication function.
    Expectation: expect correct result.
    """
    rank = get_rank()
    data = ms.Tensor(np.arange(64).reshape([8, 8]).astype(np.float32))
    out = gather_into_tensor(data, 0)
    print(out)
    if rank == 0:
        d0 = np.arange(64).reshape([8, 8]).astype(np.float32)
        gt_rank0 = np.vstack([d0, d0])
        rst = np.allclose(gt_rank0, out.asnumpy())
        assert rst
    else:
        gt_rank1 = np.array([0])
        rst = np.allclose(gt_rank1, out.asnumpy())
        assert rst

test_hccl_gather_into_tensor_func_in_cell_2p()
