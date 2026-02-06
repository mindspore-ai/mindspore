mindspore.Tensor.isclose
========================

.. py:method:: mindspore.Tensor.isclose(other, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor

    返回一个bool类型的Tensor，表示 `input` 的每个元素与 `other` 的对应元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        |input-other| <= atol + rtol * |other|

    参数：
        - **other** (Tensor) - 对比的第二个输入。
        - **rtol** (float，可选) - 相对容忍度值，默认： ``1e-05`` 。
        - **atol** (float，可选) - 绝对容忍度值，默认： ``1e-08`` 。
        - **equal_nan** (bool，可选) - 如果设置为 ``True`` ，可以认为两个 ``NaN`` 是相等的，默认值： ``False`` 。

    返回：
        Tensor。shape与广播后的shape相同，数据类型是bool。

    .. py:method:: mindspore.Tensor.isclose(x2, rtol=1e-05, atol=1e-08, equal_nan=False) -> Tensor
        :noindex:

    返回一个bool类型的Tensor，表示 `input` 的每个元素与 `x2` 的对应元素在给定容忍度内是否“接近”。其中“接近”的数学公式为：

    .. math::
        |input-x2| <= atol + rtol * |x2|

    参数：
        - **x2** (Tensor) - 对比的第二个输入。数据类型必须与 `input` 相同。
        - **rtol** (Union[float, int, bool]，可选) - 相对容忍度值，默认： ``1e-05`` 。
        - **atol** (Union[float, int, bool]，可选) - 绝对容忍度值，默认： ``1e-08`` 。
        - **equal_nan** (bool，可选) - 如果设置为 ``True`` ，可以认为两个 ``NaN`` 是相等的，默认值： ``False`` 。

    返回：
        Tensor。shape与广播后的shape相同，数据类型是bool。

    异常：
        - **TypeError** - `x2` 的类型不是Tensor。
        - **TypeError** - `input` 和 `x2` 的数据类型不在支持的类型列表中。支持的类型有float16、float32、float64、int8、int16、int32、int64和uint8，Ascend平台额外支持bfloat16和bool类型。
        - **TypeError** - `atol` 或 `rtol` 不是float、int或bool。
        - **TypeError** - `equal_nan` 不是bool。
        - **TypeError** - `input` 和 `x2` 的数据类型不同。
        - **ValueError** - `input` 和 `x2` 无法广播。
