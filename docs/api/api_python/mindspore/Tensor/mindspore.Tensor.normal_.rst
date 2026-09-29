mindspore.Tensor.normal\_
==========================

.. py:method:: mindspore.Tensor.normal_(mean=0, std=1, *, generator=None)

    使用随机数原地更新Tensor，且随机数的采样服从由参数 `mean` 和 `std` 所构成的正态分布。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **mean** (Union[bool, int, float]，可选) - 正态分布的均值，默认值： ``0``。
        - **std** (Union[bool, int, float]，可选) - 正态分布的标准差，默认值： ``1``。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`，可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。

    返回：
        返回一个Tensor，该Tensor由服从正态分布的随机数填充，且type和shape与原Tensor一致。

    异常：
        - **TypeError** - `mean` 或 `std` 的dtype不是number，即数据类型不是bool、int或float之一。
