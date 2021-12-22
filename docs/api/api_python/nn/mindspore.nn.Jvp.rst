mindspore.nn.Jvp
=================

.. py:class:: mindspore.nn.Jvp(fn)

    计算给定网络的雅可比向量积(Jacobian-vector product, JVP)。JVP对应 `前向模式自动微分 <https://mindspore.cn/docs/programming_guide/zh-CN/master/design/gradient.html#id3>`_。

    **参数：**

    - **fn** (Cell) - 基于Cell的网络，用于接收Tensor输入并返回Tensor或者Tensor元组。

    **输入：**

    - **inputs** (Tensor) - 输入网络的入参，单个或多个Tensor。
    - **v** (Tensor or Tuple of Tensor) - 与雅可比矩阵点乘的向量，Shape与网络的输入一致。

    **输出：**

    2个Tensor或Tensor元组构成的元组。

    - **net_output** (Tensor or Tuple of Tensor) - 输入网络的正向计算结果。
    - **jvp** (Tensor or Tuple of Tensor) - 雅可比向量积的结果。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> from mindspore.nn import Jvp
    >>> class Net(nn.Cell):
    ...     def construct(self, x, y):
    ...         return x**3 + y
    >>> x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    >>> v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    >>> output = Jvp(Net())(x, y, (v, v))