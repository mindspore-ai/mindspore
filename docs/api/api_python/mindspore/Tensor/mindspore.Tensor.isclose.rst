mindspore.Tensor.isclose
========================

.. py:method:: mindspore.Tensor.isclose(x2, rtol=1e-05, atol=1e-08, equal_nan=False)

    返回一个布尔型Tensor，表示当前Tensor与 `x2` 的对应元素的差异是否在容忍度内相等。

    参数：
        - **x2** (Tensor) - 对比的第二个输入，支持的类型有float32，float16，int32。
        - **rtol** (float, 可选) - 相对容忍度。默认值：1e-05。
        - **atol** (float, 可选) - 绝对容忍度。默认值：1e-08。
        - **equal_nan** (bool, 可选) - IsNan的输入，任意维度的Tensor。默认值：False。

    返回：
        Tensor，shape与广播后的shape相同，数据类型是布尔型。

    异常：
        - **TypeError** - 当前Tensor和 `x2` 中的任何一个不是Tensor。
        - **TypeError** - 当前Tensor和 `x2` 的数据类型不是float16、float32或int32之一。
        - **TypeError** - `atol` 和 `rtol` 中的任何一个不是float。
        - **TypeError** - `equal_nan`  不是bool。
        - **TypeError** - 当前Tensor和 `x2` 的数据类型不同。
        - **ValueError** - 当前Tensor和 `x2` 无法广播。
        - **ValueError** - `atol` 和 `rtol` 中的任何一个小于零。