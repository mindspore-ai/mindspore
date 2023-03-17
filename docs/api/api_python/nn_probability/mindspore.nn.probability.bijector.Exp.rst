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

        正映射，计算输入随机变量经过映射后的值，即 :math:`Y = \exp(X)`。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，输出随机变量的值。

    .. py:method:: forward_log_jacobian(value)

        计算正映射导数的对数值。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，正映射导数的对数值。

    .. py:method:: inverse(value)

        正映射，计算输出随机变量对应的输入随机变量的值，即 :math:`X = \log(Y)`。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，输入随机变量的值。

    .. py:method:: inverse_log_jacobian(value)

        计算逆映射导数的对数值。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，逆映射导数的对数值。