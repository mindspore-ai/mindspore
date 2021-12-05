mindspore.nn.probability.bijector.Exp
=======================================

.. py:class:: mindspore.nn.probability.bijector.Exp(name='Exp')

    指数Bijector（Exponential Bijector）。
    此Bijector执行如下操作：

    .. math::
        Y = \exp(x).

    **参数：**

    - **name** (str) - Bijector名称。默认值：'Exp'。

    **支持平台：**

    ``Ascend`` ``GPU``

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> from mindspore import Tensor
    >>>
    >>> # 初始化指数Bijector。
    >>> exp_bijector = nn.probability.bijector.Exp()
    >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> ans1 = exp_bijector.forward(value)
    >>> print(ans1.shape)
    (3,)
    >>> ans2 = exp_bijector.inverse(value)
    >>> print(ans2.shape)
    (3,)
    >>> ans3 = exp_bijector.forward_log_jacobian(value)
    >>> print(ans3.shape)
    (3,)
    >>> ans4 = exp_bijector.inverse_log_jacobian(value)
    >>> print(ans4.shape)
    (3,)

