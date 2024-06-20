mindspore.mint.isclose
=======================

.. py:function:: mindspore.mint.isclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个布尔型Tensor，表示 `input` 的每个元素与 `other` 的对应元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        |input-other| ≤ atol + rtol × |other|

    参数：
        - **input** (Tensor) - 对比的第一个输入，支持的类型有float16、float32、float64、int8、int16、int32、int64、uint8，Ascend平台额外支持bfloat16和bool类型。
        - **other** (Tensor) - 对比的第二个输入，数据类型必须与 `input` 相同。
        - **rtol** (Union[float, int, bool], 可选) - 相对容忍度。默认值： ``1e-05``。
        - **atol** (Union[float, int, bool], 可选) - 绝对容忍度。默认值： ``1e-08``。
        - **equal_nan** (bool, 可选) - 若为True，则两个NaN被视为相同。默认值： ``False`` 。

    返回：
        Tensor，shape与广播后的shape相同，数据类型是布尔型。

    异常：
        - **TypeError** - `input` 和 `other` 中的任何一个不是Tensor。
        - **TypeError** - `input` 和 `other` 的数据类型不在支持的类型列表中。
        - **TypeError** - `atol` 和 `rtol` 中的任何一个不是float、int或bool。
        - **TypeError** - `equal_nan` 不是bool。
        - **TypeError** - `input` 和 `other` 的数据类型不同。
        - **ValueError** - `input` 和 `other` 无法广播。
