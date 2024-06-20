mindspore.ops.IsClose
=====================

.. py:class:: mindspore.ops.IsClose(rtol=1e-05, atol=1e-08, equal_nan=True)

    返回一个bool型Tensor，表示 `input` 的每个元素与 `other` 的每个元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        |input-other| ≤ atol + rtol × |other|

    更多参考详见 :func:`mindspore.ops.isclose`。

    参数：
        - **rtol** (float, 可选) - 相对容忍度。默认值： ``1e-05`` 。
        - **atol** (float, 可选) - 绝对容忍度。默认值： ``1e-08`` 。
        - **equal_nan** (bool, 可选) - 若为True，则两个NaN被视为相同。默认值： ``True`` 。

    输入：
        - **input** (Tensor) - 对比的第一个输入。
        - **other** (Tensor) - 对比的第二个输入。

    输出：
        Tensor，shape与 `input` 和 `other` 广播后的shape相同，数据类型是bool。
