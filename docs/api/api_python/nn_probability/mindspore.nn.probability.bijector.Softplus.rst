mindspore.nn.probability.bijector.Softplus
=================================================

.. py:class:: mindspore.nn.probability.bijector.Softplus(sharpness=1.0, name='Softplus')

    Softplus Bijector。
    此Bijector对应的映射函数为：

    .. math::
        Y = g(x) = \frac{\log(1 + e ^ {kX})}{k}

    其中k是锐度因子。

    **参数：**

    - **sharpness** (float, list, numpy.ndarray, Tensor) - 锐度因子，上述公式中的k。默认值：1.0。
    - **name** (str) - Bijector名称。默认值：'Softplus'。

    **支持平台：**

    ``Ascend`` ``GPU``

    .. note::
        `sharpness` 中元素的数据类型必须为float。

    **异常：**

    - **TypeError** - sharpness中元素的数据类型不为float。

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

    .. py:method:: forward(value)

        正映射，计算输入随机变量 :math:`X = value` 经过映射后的值 :math:`Y = g(value)`。

        **参数：**

        - **value** (Tensor) - 输入随机变量的值。

        **返回：**

        Tensor, 输入随机变量的值。

    .. py:method:: forward_log_jacobian(value)

        计算正映射导数的对数值，即 :math:`\log(dg(x) / dx)`。

        **参数：**

        - **value** (Tensor) - 输入随机变量的值。

        **返回：**

        Tensor, 正映射导数的对数值。

    .. py:method:: inverse(value)

        正映射，计算输出随机变量 :math:`Y = value` 时对应的输入随机变量的值 :math:`X = g(value)`。

        **参数：**

        - **value** (Tensor) - 输出随机变量的值。

        **返回：**

        Tensor, 输出随机变量的值。

    .. py:method:: inverse_log_jacobian(value)

        计算逆映射导数的对数值，即 :math:`\log(dg^{-1}(x) / dx)`。

        **参数：**

        - **value** (Tensor) - 输出随机变量的值。

        **返回：**

        Tensor, 逆映射导数的对数值。
