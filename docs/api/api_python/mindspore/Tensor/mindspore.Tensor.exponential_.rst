mindspore.Tensor.exponential\_
===============================

.. py:method:: mindspore.Tensor.exponential_(lambd=1, *, generator=None)

    根据指数分布生成随机数填充Tensor。

    .. math::
        f(x) = \lambda \exp(-\lambda x)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **lambd** (float, 可选) - 指数分布的参数。默认值： ``1`` 。

    关键字参数：
        - **generator** (Generator, 可选) - 随机数生成器。默认值： ``None`` 。

    返回：
        Tensor，shape和数据类型与输入相同。
