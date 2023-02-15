mindspore.ops.full_like
=======================

.. py:function:: mindspore.ops.full_like(x, fill_value, *, dtype=None)

    返回一个shape与 `x` 相同并且使用 `fill_value` 填充的Tensor。

    参数：
        - **x** (Tensor) - 输入Tensor，输出Tensor与 `x` 具有相同的shape。
        - **fill_value** (number.Number) - 用来填充输出Tensor的值。当前不支持复数类型。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 指定输出Tensor的数据类型。数据类型只支持 `bool_` 和 `number` ，更多细节详见 :class:`mindspore.dtype` 。默认值：None。

    返回：
        Tensor。

    异常：
        - **TypeError** - `x` 不是Tensor。
