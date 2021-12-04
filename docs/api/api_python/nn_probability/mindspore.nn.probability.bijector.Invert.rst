mindspore.nn.probability.bijector.Invert
============================================

.. py:class:: mindspore.nn.probability.bijector.Invert(bijector, name='')

    反转Bijector（Invert Bijector），计算输入Bijector的反函数。

    **参数：**

    - **Bijector** (Bijector) - 基础Bijector（Base Bijector）。
    - **name** (str) - Bijector名称。默认值：""。当name设置为""时，它实际上是'Invert' + Bijector.name。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.bijector as msb
    >>> from mindspore import Tensor
    >>> class Net(nn.Cell):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
    ...         self.invert = msb.Invert(self.origin)
    ...
    ...     def construct(self, x_):
    ...         return self.invert.forward(x_)
    >>> forward = Net()
    >>> x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    >>> ans = forward(Tensor(x, dtype=mindspore.float32))
    >>> print(ans.shape)
    (4,)

    .. py:method:: bijector
        :property:

        返回基础Bijector。

    .. py:method:: forward(x)

        逆变换：将输入值转换回原始分布。

        **参数：**

        - **x** (Tensor) - 输入。

    .. py:method:: forward_log_jacobian(x)

        逆变换导数的对数。

        **参数：**

        - **x** (Tensor) - 输入。

    .. py:method:: inverse(y)

        正变换：将输入值转换为另一个分布。

        **参数：**

        - **y** (Tensor) - 输入。

    .. py:method:: inverse_log_jacobian(y)

        正变换导数的对数。

        **参数：**

        - **y** (Tensor) - 输入。

