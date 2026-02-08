mindspore.mint.allclose
=======================

.. py:function:: mindspore.mint.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回 `input` 中的所有元素与 `other` 的对应元素是否“接近”。其中“接近”的数学公式为：

    .. math::
        |input-other| ≤ atol + rtol × |other|

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。
        - **rtol** (Union[float, int, bool], 可选) - 相对容忍度。默认 ``1e-05``。
        - **atol** (Union[float, int, bool], 可选) - 绝对容忍度。默认 ``1e-08``。
        - **equal_nan** (bool, 可选) - 两个NaN是否被视为相同。默认 ``False`` 。

    返回：
        布尔型标量。
