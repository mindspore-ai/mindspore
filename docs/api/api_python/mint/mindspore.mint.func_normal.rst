mindspore.mint.normal
=======================

.. py:function:: mindspore.mint.normal(mean=0.0, std=1.0, size=None, generator=None)

    根据正态（高斯）随机数分布生成随机数。

    参数：
        - **mean** (Union[float, Tensor]，可选) - 每个元素的均值, shape与std相同。默认值： ``0.0``。
        - **std** (Union[float, Tensor]，可选) - 每个元素的标准差， shape与mean相同。std的值大于等于0。默认值： ``1.0``。
        - **size** (tuple，可选) - 当mean和std为常量时，指定输出shape。默认值:  ``None``。
        - **generator** (generator，可选) - MindSpore随机种子。默认值： ``None``。

    返回：
        Tensor，输出tensor的shape和mean的shape相同，或者在mean和std为常量时，shape为size。

    异常：
        - **TypeError** - 如果 `mean` 或 `std` 不是Union[float, Tensor]。
