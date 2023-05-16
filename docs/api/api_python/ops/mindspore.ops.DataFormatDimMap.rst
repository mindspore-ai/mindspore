mindspore.ops.DataFormatDimMap
==============================

.. py:class:: mindspore.ops.DataFormatDimMap(src_format='NHWC', dst_format='NCHW')

    返回源数据格式中的目标数据格式的维度索引。

    参数：
        - **src_format** (str) - 源数据格式中的可选值。格式可以是 ``'NHWC'`` 和 ``'NCHW'`` 。默认值： ``'NHWC'`` 。
        - **dst_format** (str) - 目标数据格式中的可选值。格式可以是 ``'NHWC'`` 和 ``'NCHW'`` 。默认值： ``'NCHW'`` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor，每个元素都用作源数据格式的维度索引。建议值在[-4, 4)范围内，仅支持int32。

    输出：
        输出Tensor，返回给定目标数据格式的维度索引，与 `input_x` 具有相同的数据类型和shape。

    异常：
        - **TypeError** - `src_format` 或 `dst_format` 不是str。
        - **TypeError** - `input_x` 不是数据类型为int32的Tensor。