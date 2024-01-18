import mindspore as ms
import numpy as np

ms.set_context(mode=ms.GRAPH_MODE)


class MatMulCustom(ms.nn.Cell):
    def __init__(self, ta=False, tb=True):
        super().__init__()
        self.net = ms.ops.MatMul(ta,tb)
    
    def construct(self, i0, i1):
        return self.net(i0,i1)

def run_transB(m, k, n, mstype):
    i0_host = np.random.random([m,k]).astype(np.float16)
    i1_host = np.random.random([n,k]).astype(np.float16)
    
    i0_tensor = ms.Tensor(i0_host, ms.float16)
    i1_tensor = ms.Tensor(i1_host, ms.float16)

    expect = np.matmu(i0_host.astype(np.float32), i1_host.astype(np.float32).T).astype(np.float16)
    matmul = MatMulCustom(False, True)
    output = matmul(i0_tensor, i1_tensor)
    assert (np.abs(expect - output) < 1e-1).all()
    print("run matmul transB success, Shape = ", m,k,n,mstype)

if __name__ == "__main__":
    run_transB(1, 11264, 136960)
    run_transB(1024, 11264, 1664)
    run_transB(1024, 11264, 6912)
    run_transB(1024, 1408, 11264)
    run_transB(1024, 16912, 11264)
    run_transB(4, 11264, 136960)
    run_transB(4, 11264, 1664)
    run_transB(4, 11264, 6912)
    run_transB(4, 1408, 11264)
    run_transB(4, 6912, 11264)
