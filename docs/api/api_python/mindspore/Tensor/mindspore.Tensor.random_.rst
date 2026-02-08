mindspore.Tensor.random\_
=======================================

.. py:method:: mindspore.Tensor.random_(from_=0, to=None, *, generator=None)

    在区间 :math:`[from\_, to-1]` 内生成服从离散均匀分布的随机数，原地更新输入Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **from_** (Union[numbers.Number, Tensor], 可选) - 均匀分布的下界，可以是一个标量值或只有单个元素的任意维度的Tensor，默认值： ``0``。
        - **to** (Union[numbers.Number, Tensor], 可选) - 均匀分布的上界，可以是一个标量值或只有单个元素的任意维度的Tensor。默认为输入数据类型的取值上限。默认值： ``None``。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`，可选) - 伪随机数生成器。默认值： ``None`` ，使用默认伪随机数生成器。

    返回：
        返回输入tensor。

    异常：
        - **TypeError** - `from_` 或 `to` 不是整型。
        - **RuntimeError** - 如果 `from_` 大于等于 `to`。