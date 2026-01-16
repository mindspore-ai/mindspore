mindspore.ops.isclose
=====================

.. py:function:: mindspore.ops.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个布尔型tensor，表示两个tensor在容忍度内是否逐元素相等。数学公式为：

    .. math::
        |input-other| ≤ atol + rtol × |other|

    如果inf值具有相同的符号，则认为它们相等；如果 `equal_nan` 为 ``True`` ，NaN值被认为相等。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。
        - **rtol** (Union[float, int, bool], 可选) - 相对容忍度。默认 ``1e-05`` 。
        - **atol** (Union[float, int, bool], 可选) - 绝对容忍度。默认 ``1e-08`` 。
        - **equal_nan** (bool, 可选) - 两个NaN是否被视为相等。默认 ``False`` 。

    返回：
        Tensor
