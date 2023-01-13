mindspore.ops.pinv
=========================

.. py:function:: mindspore.ops.pinv(x, *, atol=None, rtol=None, hermitian=False)

    计算矩阵的（Moore-Penrose）伪逆。

    参数：
        - **x** (Tensor) - 要计算的矩阵。矩阵必须至少有两个维度。支持数据类型为float32或float64。
          
          - hermitian为false时，支持2维和更高维度，shape为 :math:`(*, M, N)`。
          - hermitian为true时，仅支持2维，shape为 :math:`(M, N)`。

    关键字参数：
        - **atol** (float, Tensor) - 绝对公差值。默认值：None。
        - **rtol** (float, Tensor) - 相对公差值。默认值：None。
        - **hermitian** (bool) - 为True时假设x为实对称矩阵。默认值：False。

    输出：
        - **output** (Tensor): 类型与输入相同。
          
          - hermitian为false时，输出shape为 :math:`(*, N, M)`。
          - hermitian为true时，输出shape为 :math:`(N, M)`。

    异常：
        - **TypeError** - `hermitian` 不是bool。
        - **TypeError** - `x` 不是Tensor。
        - **ValueError** - `x` 的维度小于2。