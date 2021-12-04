mindspore.nn.Vjp
=================

.. py:class:: mindspore.nn.Vjp(fn)

    计算给定网络的向量雅可比积(vector-Jacobian product, VJP)。VJP对应反向模式自动微分。

    **参数：**

    - **fn** (Cell) - 基于Cell的网络，用于接收张量输入并返回张量或者张量元组。

    **输入：**

    - **inputs** (Tensor) - 输入网络的入参，单个或多个张量。
    - **v** (Tensor or Tuple of Tensor) - 与雅可比矩阵点乘的向量，形状与网络的输出一致。

    **输出：**

    2个张量或张量元组构成的元组。

    - **net_output** (Tensor or Tuple of Tensor) - 输入网络的正向计算结果。
    - **vjp** (Tensor or Tuple of Tensor) - 向量雅可比积的结果。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> from mindspore.nn import Vjp
    >>> class Net(nn.Cell):
    ...     def construct(self, x, y):
    ...         return x**3 + y
    >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    >>> output = Vjp(Net())(x, y, v)