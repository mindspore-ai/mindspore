mindspore.Tensor.uniform\_
=======================================

.. py:method:: mindspore.Tensor.uniform_(from_=0, to=1, *, generator=None)

    通过在半开区间 :math:`[from\_, to)` 内生成服从均匀分布的随机数来原地更新输入Tensor。

    .. math::
        P(x) = \frac{1}{to - from\_}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **from_** (Union[number.Number, Tensor], 可选) - 均匀分布的下界，可以是一个标量值或只有单个元素的任意维度的Tensor，默认值： ``0``。
        - **to** (Union[number.Number, Tensor], 可选) - 均匀分布的上界，可以是一个标量值或只有单个元素的任意维度的Tensor，默认值： ``1``。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。

    返回：
        返回输入tensor。

    异常：
        - **TypeError** - `from_` 或 `to` 既不是number也不是Tensor。
        - **TypeError** - `from_` 或 `to` 为Tensor类型，但数据类型不是bool、int8、int16、int32、int64、uint8、float32、float64之一。
        - **ValueError** - `from_` 或 `to` 为Tensor类型，但是有多个元素。
        - **RuntimeError** - 如果 `from_` 大于 `to`。
