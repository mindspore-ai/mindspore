mindspore.ops.GridSampler2D
===========================

.. py:class:: mindspore.ops.GridSampler2D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=False)

    此操作使用基于流场网格的插值对 2D input_x进行采样，该插值通常由 :func:`mindspore.ops.affine_grid` 生成。

    参数：
        - **interpolation_mode** (str，可选) - 指定插值方法的可选字符串。可选值为："bilinear"、"nearest"，默认为："bilinear"。
        - **padding_mode** (str，可选) - 指定填充模式的可选字符串。可选值为："zeros"、"border"或者"reflection"，默认为："zeros"。
        - **align_corners** (bool，可选) - 一个可选bool。如果为True，输入和输出Tensor的角像素的中心是对齐的，如果为False，
          则不是对齐的。默认为：False。

    输入：
        - **input_x** (Tensor) - 一个4D的Tensor，dtype为float16或者float32，shape为 :math:`(N, C, H_{in}, W_{in})` 。
        - **grid** (Tensor) - 一个4D的Tensor，dtype和 `input_x` 相同，shape为 :math:`(N, H_{out}, W_{out}, 2)` ，
          用于指定由输入空间维度归一化的采样像素位置。

    输出：
        一个4-D的Tensor，dtype和 `input_x` 相同，shape为 :math:`(N, C, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - 如果 `input_x` 或者 `grid` 不是一个Tensor。
        - **TypeError** - 如果 `input_x` 和 `grid` 的dtype不兼容。
        - **TypeError** - 如果 `input_x` 或者 `grid` 的dtype无效。
        - **TypeError** - 如果 `align_corners` 不是一个布尔值。
        - **ValueError** - 如果 `input_x` 或者 `grid` 的维度不等于4。
        - **ValueError** - 如果 `input_x` 的第一根轴和 `grid` 的不相等。
        - **ValueError** - 如果 `grid` 的第四根轴不等于2。
        - **ValueError** - 如果 `interpolation_mode` 不是"bilinear"或者"nearest"的字符串。
        - **ValueError** - 如果 `padding_mode` 不是"zeros"、"border"、"reflection"的字符串。
