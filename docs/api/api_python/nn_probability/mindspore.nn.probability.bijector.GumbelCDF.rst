mindspore.nn.probability.bijector.GumbelCDF
============================================

.. py:class:: mindspore.nn.probability.bijector.GumbelCDF(loc=0.0, scale=1.0, name='GumbelCDF')

    GumbelCDF Bijector。
    此Bijector对应的映射函数为：

    .. math::
        Y = g(x) = \exp(-\exp(\frac{-(X - loc)}{scale}))

    参数：
        - **loc** (float, list, numpy.ndarray, Tensor) - 位移因子，即上述公式中的loc。默认值：0.0。
        - **scale** (float, list, numpy.ndarray, Tensor) - 比例因子，即上述公式中的scale。默认值：1.0。
        - **name** (str) - Bijector名称。默认值：'GumbelCDF'。

    .. note::
        - `scale` 中元素必须大于零。
        - 对于 `inverse` 和 `inverse_log_jacobian` ，输入应在(0, 1)范围内。
        - `loc` 和 `scale` 中元素的数据类型必须为float。
        - 如果 `loc` 、 `scale` 作为numpy.ndarray或Tensor传入，则它们必须具有相同的数据类型，否则将引发错误。

    异常：
        - **TypeError** - `loc` 或 `scale` 中元素的数据类型不为float，或 `loc` 和 `scale` 中元素的数据类型不相同。

    .. py:method:: loc
        :property:

        返回分布位置。

        返回：
            Tensor，分布的位置值。

    .. py:method:: scale
        :property:

        返回分布比例。

        返回：
            Tensor，分布的比例值。

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
