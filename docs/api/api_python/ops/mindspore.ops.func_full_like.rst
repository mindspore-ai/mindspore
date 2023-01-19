mindspore.ops.full_like
=======================

.. py:function:: mindspore.ops.full_like(x, fill_value, *, dtype=None)

    返回一个与输入相同大小的Tensor，填充 `fill_value`。'ops.full_like(x, fill_value)'相当于'ops.full(x.shape, fill_value, dtype=x.dtype)'。

    参数：
        - **x** (Tensor) - `x` 的shape决定输出Tensor的shape。
        - **fill_value** (number.Number) - 用来填充输出Tensor的值。当前不支持复数类型。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定输出Tensor的数据类型。数据类型只支持 `bool_` 和 `number` ，更多细节详见 :class:`mindspore.dtype` 。默认值：None。

    返回：
        Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
