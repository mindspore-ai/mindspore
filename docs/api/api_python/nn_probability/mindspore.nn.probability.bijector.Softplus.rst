mindspore.nn.probability.bijector.Softplus
=================================================

.. py:class:: mindspore.nn.probability.bijector.Softplus(sharpness=1.0, name='Softplus')

    Softplus Bijector。
    此Bijector执行如下操作：

    .. math::
        Y = \frac{\log(1 + e ^ {kX})}{k}

    其中k是锐度因子。

    **参数：**

    - **sharpness** (float, list, numpy.ndarray, Tensor) - 锐度因子，上述公式中的k。默认值：1.0。
    - **name** (str) - Bijector名称。默认值：'Softplus'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        `sharpness` 的数据类型必须为float。

    **异常：**

    - **TypeError** - sharpness的数据类型不为float。

    **样例：**

    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.nn.probability.bijector as msb
    >>> from mindspore import Tensor
    >>>
    >>> # 初始化Softplus Bijector，sharpness设置为2.0。
    >>> softplus = msb.Softplus(2.0)
    >>> # 在网络中使用ScalarAffine Bijector。
    >>> value = Tensor([1, 2, 3], dtype=mindspore.float32)
    >>> ans1 = softplus.forward(value)
    >>> print(ans1.shape)
    (3,)
    >>> ans2 = softplus.inverse(value)
    >>> print(ans2.shape)
    (3,)
    >>> ans3 = softplus.forward_log_jacobian(value)
    >>> print(ans3.shape)
    (3,)
    >>> ans4 = softplus.inverse_log_jacobian(value)
    >>> print(ans4.shape)
    (3,)

