mindspore.ops.standard_normal
=============================

.. py:function:: mindspore.ops.standard_normal(shape, seed=None)

    根据标准正态（高斯）随机数分布生成随机数。

    .. math::
        f(x)=\frac{1}{\sqrt{2 \pi}} e^{\left(-\frac{x^{2}}{2}\right)}

    .. warning::
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        - 该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **shape** (Union[tuple, Tensor]) - 指定生成随机数的shape。
        - **seed** (int, 可选) - 随机数种子。默认 ``None`` 。

    返回：
        Tensor

    异常：
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
        - **ValueError** - `shape` 为秩不等于1的tensor。
