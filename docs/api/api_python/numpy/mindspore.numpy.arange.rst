mindspore.numpy.arange
=================================

.. py:function:: mindspore.numpy.arange(start, stop=None, step=None, dtype=None)

    返回给定区间内均匀间隔的值。

    参数：
        - **start** (Union[int, float]) - 区间的起始值，区间包含该值。如果希望以位置参数处理入参 `stop` ，那么 `start` 是必选参数；否则， `start` 是可选参数，默认值： ``0`` 。
        - **stop** (Union[int, float], 可选) - 用于指定生成的数组中的终止值，不包括该值本身（除非在某些情况下 ``step`` 不是整数并且浮点数的舍入会影响输出的数组长度）。
        - **step** (Union[int, float], 可选) - 用于指定数组中连续的两个值之间的间距。如果未指定 ``step`` 参数，数组中的元素间距默认值： ``1`` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 如果指定了该参数，则生成数组的数据类型将采用给定的数据类型。如果未指定，则生成数组的数据类型将根据提供的参数自动推断出来。默认值： ``None`` 。

    返回：
        Tensor，给定区间范围内以均匀间隔生成的值。

    异常：
        - **TypeError (PyNative Mode)** - 如果输入的参数类型没有按上述内容指定，或者没有按照上述内容给定的顺序输入参数。
        - **RuntimeError (Graph Mode)** - 在PyNative模式中导致TypeError的输入将在Graph模式中导致RuntimeError。