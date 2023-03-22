mindspore.nn.probability.bijector.PowerTransform
=================================================

.. py:class:: mindspore.nn.probability.bijector.PowerTransform(power=0., name='PowerTransform')

    乘方Bijector（PowerTransform Bijector）。
    此Bijector对应的映射函数为：

    .. math::
        Y = g(X) = (1 + X * c)^{1 / c}, X >= -1 / c

    其中c >= 0。

    PowerTransform Bijector将输入从 `[-1/c, inf]` 映射到 `[0, inf]` 。

    当 `c=0` 时，此Bijector等于 :class:`mindspore.nn.probability.bijector.Exp` Bijector。

    参数：
        - **power** (float, list, numpy.ndarray, Tensor) - 比例因子。默认值：0。
        - **name** (str) - Bijector名称。默认值：'PowerTransform'。

    .. note::
        `power` 中元素的数据类型必须为float。

    异常：
        - **ValueError** - `power` 中元素小于0或静态未知。
        - **TypeError** - `power` 中元素的数据类型不是float。

    .. py:method:: power
        :property:

        返回指数。

        返回：
            Tensor，Bijector的指数。

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
