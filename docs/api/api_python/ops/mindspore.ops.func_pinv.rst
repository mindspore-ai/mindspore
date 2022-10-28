mindspore.ops.pinv
=========================

.. py:function:: mindspore.ops.pinv(x, *, atol=None, rtol=None, hermitian=False)

    计算矩阵的（Moore-Penrose）伪逆。

    参数：
        - **x** (Tensor) - 要计算的矩阵。矩阵必须至少有两个维度。支持数据类型为float32或float64。

    关键字参数：
        - **atol** (float, tensor) - 绝对公差值。默认值：None。
        - **rtol** (float, tensor) - 相对公差值。默认值：None。
        - **hermitian** (bool) - 为True时假设x为实对称矩阵。默认值：False。

    输出：
        Tensor，类型与输入相同。

    异常：
        - **TypeError** - `hermitian` 不是bool。
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 的维度小于2。