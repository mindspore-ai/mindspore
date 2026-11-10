mindspore.ops.standard_laplace
==============================

.. py:function:: mindspore.ops.standard_laplace(shape, seed=None)

    根据标准Laplace（mean=0, lambda=1）分布生成随机数。

    .. math::
        \text{f}(x) = \frac{1}{2}\exp(-|x|)

    .. warning::
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。
        - 本版本已移除 ``mindspore.nn.probability.distribution.Laplace`` 所属的概率模块。
          此前关于将本接口改为分布对象形式、并迁移至 ``Laplace`` 的指引不再适用。
          本接口当前仍保留函数形式及现有参数，可继续用于标准Laplace分布采样；
          这不代表其提供原 ``Laplace`` 分布类的全部功能。

    参数：
        - **shape** (Union[tuple, Tensor]) - 指定生成随机数的shape。
        - **seed** (int, 可选) - 随机数种子。默认 ``None`` 。

    返回：
        Tensor

    异常：
        - **ValueError** - `shape` 为tuple时，包含非正的元素。
        - **ValueError** - `shape` 为秩不等于1的tensor。
