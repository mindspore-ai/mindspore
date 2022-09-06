mindspore.nn.probability.bijector.Invert
============================================

.. py:class:: mindspore.nn.probability.bijector.Invert(bijector, name='')

    逆映射Bijector（Invert Bijector）。
    计算输入Bijector的逆映射。如果正向映射（下面的 `bijector` 输入）对应的映射函数为 :math:`Y = g(X)` ，那么对应的逆映射Bijector的映射函数为 :math:`Y = h(X) = g^{-1}(X)` 。

    参数：
        - **bijector** (Bijector) - 基础Bijector（Base Bijector）。
        - **name** (str) - Bijector名称。默认值：""。当name设置为""时，实际上是'Invert' + Bijector.name。

    .. py:method:: bijector
        :property:

        Bijector类，返回基础Bijector。

    .. py:method:: forward(x)

        计算基础Bijector的逆映射，即 :math:`Y = h(X) = g^{-1}(X)` 。

        参数：
            - **x** (Tensor) - 基础Bijector的输出随机变量的值。

        返回：
            Tensor，基础Bijector的输入随机变量的值。

    .. py:method:: forward_log_jacobian(x)

        计算基础Bijector的逆映射导数的对数值，即 :math:`\log dg^{-1}(x) / dx` 。

        参数：
            - **x** (Tensor) - 基础Bijector的输出随机变量的值。

        返回：
            Tensor，基类逆映射导数的对数值。

    .. py:method:: inverse(y)

        计算基础Bijector的正映射，即 :math:`Y = g(X)` 。

        参数：
            - **y** (Tensor) - 基础Bijector的输入随机变量的值。

        返回：
            Tensor，基础Bijector的输出随机变量的值。

    .. py:method:: inverse_log_jacobian(y)

        计算基础Bijector的正映射导数的对数，即 :math:`Y = \log dg(x) / dx` 。

        参数：
            - **y** (Tensor) - 基础Bijector的输入随机变量的值。

        返回：
            Tensor，基类正映射导数的对数值。

