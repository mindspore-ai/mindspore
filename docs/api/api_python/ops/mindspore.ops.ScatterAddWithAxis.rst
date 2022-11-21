mindspore.ops.ScatterAddWithAxis
=================================

.. py:class:: mindspore.ops.ScatterAddWithAxis(axis=0)

    该操作的输出是通过创建输入 `input_x` 的副本，然后将 `updates` 指定的值添加到 `indices` 指定的位置来更新副本中的值。

    .. note::
        三个输入 `input_x`, `updates` 和 `indices` 的秩相同且都大于等于1。

    参数：
        - **axis** (int，可选) - 指定在哪个轴上进行散点加法。默认值：0。

    输入：
        - **input_x** (Parameter) - 相加操作目标Tensor。
        - **indices** (Tensor) - 指定相加操作的索引，数据类型为int32或者int64。
        - **updates** (Tensor) - 指定与 `input_x` 相加操作的Tensor，数据类型与 `input_x` 相同，shape与 `indices` 相同。

    输出：
        Tensor，更新后的 `input_x` ，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `indices` 不是int32或者int64。
        - **ValueError** - `indices` 和 `updates` 的shape不一致。
