mindspore.nn.probability.bijector.ScalarAffine
=================================================

.. py:class:: mindspore.nn.probability.bijector.ScalarAffine(scale=1.0, shift=0.0, name='ScalarAffine')

    标量仿射Bijector（Scalar Affine Bijector）。
    此Bijector对应的映射函数为：

    .. math::
        Y = a * X + b

    其中a是比例因子，b是移位因子。

    参数：
        - **scale** (float, list, numpy.ndarray, Tensor) - 比例因子。默认值：1.0。
        - **shift** (float, list, numpy.ndarray, Tensor) - 移位因子。默认值：0.0。
        - **name** (str) - Bijector名称。默认值：'ScalarAffine'。

    .. note::
        `shift` 和 `scale` 中元素的数据类型必须为float。如果 `shift` 、 `scale` 作为numpy.ndarray或Tensor传入，则它们必须具有相同的数据类型，否则将引发错误。

    异常：
        - **TypeError** - `shift` 或 `scale` 中元素的数据类型不为float，或 `shift` 和 `scale` 的数据类型不相同。

    .. py:method:: shift
        :property:

        返回映射的位置。

        返回：
            Tensor，映射的位置值。

    .. py:method:: scale
        :property:

        返回映射的比例。

        返回：
            Tensor，映射的比例值。

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
