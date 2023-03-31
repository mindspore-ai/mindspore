mindspore.ops.Pad
==================

.. py:class:: mindspore.ops.Pad(paddings)

    根据参数 `paddings` 对输入进行填充。

    更多参考详见 :func:`mindspore.ops.pad`。如果 `paddings` 里存在负数值，请使用 :func:`mindspore.ops.pad` 接口。

    参数：
        - **paddings** (tuple) - 填充大小，其shape为(N, 2)，N是输入数据的维度，填充的元素为int类型。对于 `x` 的第 `D` 个维度，paddings[D, 0]表示输入Tensor的第 `D` 维度前面要扩展的大小，paddings[D, 1]表示在输入Tensor的第 `D` 个维度后面要扩展的大小。

    输入：
        - **input_x** (Tensor) - 被填充的Tensor，其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。

    输出：
        填充后的Tensor。

    异常：
        - **TypeError** - `paddings` 不是tuple。
        - **TypeError** - `input_x` 不是Tensor。
        - **ValueError** - `paddings` 的shape不是 :math:`(N, 2)` 。
        - **ValueError** - `paddings` 的大小不等于2 * len(input_x)。