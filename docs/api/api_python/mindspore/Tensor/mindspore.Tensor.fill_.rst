mindspore.Tensor.fill\_
=========================

.. py:method:: mindspore.Tensor.fill_(value) -> Tensor

    用指定的值填充 `self` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **value** (Union[Tensor, number.Number, bool]) - 用来填充 `self` 的值。

    返回：
        Tensor。

    异常：
        - **RuntimeError** - `self` 或 `value` 的数据类型不支持。
        - **RuntimeError** - 当 `value` 是Tensor时，它应该是0-D Tensor或shape=[1]的1-D Tensor。