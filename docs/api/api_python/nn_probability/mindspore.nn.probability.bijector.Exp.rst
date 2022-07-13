mindspore.nn.probability.bijector.Exp
=======================================

.. py:class:: mindspore.nn.probability.bijector.Exp(name='Exp')

    指数Bijector（Exponential Bijector）。
    此Bijector对应的映射函数为：

    .. math::
        Y = \exp(x).

    参数：
        - **name** (str) - Bijector名称。默认值：'Exp'。


    .. py:method:: forward(value)

        正映射，计算输入随机变量 :math:`X = value` 经过映射后的值 :math:`Y = \exp(value)`。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，输出随机变量的值。

    .. py:method:: forward_log_jacobian(value)

        计算正映射导数的对数值，即 :math:`\log(d\exp(x) / dx)`。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，正映射导数的对数值。

    .. py:method:: inverse(value)

        正映射，计算输出随机变量 :math:`Y = value` 时对应的输入随机变量的值 :math:`X = \log(value)`。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，输入随机变量的值。

    .. py:method:: inverse_log_jacobian(value)

        计算逆映射导数的对数值，即 :math:`\log(d\log(x) / dx)`。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，逆映射导数的对数值。