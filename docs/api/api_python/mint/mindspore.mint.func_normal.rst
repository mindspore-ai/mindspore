mindspore.mint.normal
=======================

.. py:function:: mindspore.mint.normal(mean, std, *, generator=None) -> Tensor

    根据正态（高斯）随机数分布生成随机数。

    参数：
        - **mean** (Union[Tensor]) - 每个元素的均值，shape与 `std` 相同。
        - **std** (Union[Tensor]) - 每个元素的标准差，shape与 `mean` 相同。 `std` 的值大于等于0。

    关键字参数：
        - **generator** (generator，可选) - MindSpore随机种子。默认值： ``None``。

    返回：
        Tensor，输出tensor的shape和 `mean` 的shape相同。

    异常：
        - **TypeError** - 如果 `mean` 或 `std` 不是Union[float, Tensor]。

    .. py:function:: mindspore.mint.normal(mean, std) -> Tensor
        :noindex:

    与上面接口类似。但所有生成的元素共享同一个均值。

    参数：
        - **mean** (float) - 每个元素的均值。
        - **std** (Tensor) - 每个元素的标准差。 `std` 的值大于等于0。

    返回：
        Tensor，输出tensor的shape和 `std` 的shape相同。

    .. py:function:: mindspore.mint.normal(mean, std=1.0) -> Tensor
        :noindex:

    与上面接口类似。但所有生成的元素共享同一个标准差。

    参数：
        - **mean** (Tensor) - 每个元素的均值。
        - **std** (float，可选) - 每个元素的标准差。 `std` 的值大于等于0。默认值： ``1.0``。

    返回：
        Tensor，输出tensor的shape和 `mean` 的shape相同。

    .. py:function:: mindspore.mint.normal(mean, std, size) -> Tensor
        :noindex:

    与上面接口类似。但所有生成的元素共享同一个均值和标准差。输出Tensor的大小由 `size` 指定。

    参数：
        - **mean** (float) - 每个元素的均值。
        - **std** (float) - 每个元素的标准差。
        - **size** (tuple) - 输出shape。

    返回：
        Tensor，shape为 `size` 。