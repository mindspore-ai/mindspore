mindspore.nn.probability.bijector.Softplus
=================================================

.. py:class:: mindspore.nn.probability.bijector.Softplus(sharpness=1.0, name='Softplus')

    Softplus Bijector。
    此Bijector对应的映射函数为：

    .. math::
        Y = g(x) = \frac{\log(1 + e ^ {kX})}{k}

    其中k是锐度因子。

    参数：
        - **sharpness** (float, list, numpy.ndarray, Tensor) - 锐度因子，上述公式中的k。默认值：1.0。
        - **name** (str) - Bijector名称。默认值：'Softplus'。

    .. note::
        `sharpness` 中元素的数据类型必须为float。

    异常：
        - **TypeError** - sharpness中元素的数据类型不为float。

    .. py:method:: sharpness
        :property:

        返回映射的锐度因子。

        返回：
            Tensor，映射的锐度因子。

    .. py:method:: forward(value)

        正映射，计算输入随机变量经过映射后的值。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，输入随机变量的值。

    .. py:method:: forward_log_jacobian(value)

        计算正映射导数的对数值。

        参数：
            - **value** (Tensor) - 输入随机变量的值。

        返回：
            Tensor，正映射导数的对数值。

    .. py:method:: inverse(value)

        正映射，计算输出随机变量对应的输入随机变量的值。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，输出随机变量的值。

    .. py:method:: inverse_log_jacobian(value)

        计算逆映射导数的对数值。

        参数：
            - **value** (Tensor) - 输出随机变量的值。

        返回：
            Tensor，逆映射导数的对数值。
