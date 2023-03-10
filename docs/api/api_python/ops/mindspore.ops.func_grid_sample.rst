mindspore.ops.grid_sample
=========================

.. py:function:: mindspore.ops.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。`input` 只支持4-D（GridSampler2D）和5-D（GridSampler3D）。

    在4-D场景下，`input` 的shape为 :math:`(N, C, H_{in}, W_{in})`，`grid` 的shape为 :math:`(N, H_{out}, W_{out}, 2)`，`output` 的shape为 :math:`(N, C, H_{out}, W_{out})`。
    对于每个输出位置 `output[n, :, h, w]`，`grid[n, h, w]` 指定 `input` 像素位置 `x` 和 `y`，用于计算 `output[n, :, h, w]` 的插值。以5D为例，`grid[n, d, h, w]` 指定 `x`，
    `y`，`z` 像素位置的插值位置为[n, :, d, h, w]。`mode` 参数指定 `nearest` 或 `bilinear` 或 `bicubic` (仅支持4D情况)插值法对输入像素进行采样。

    `grid` 指定由 `input` 归一化的采样像素位置。因此，它应该在 :math:`[-1, 1]` 范围内的值最多。

    如果 `grid` 的值在 :math:`[-1, 1]` 范围之外，则相应的输出将按照定义的 `padding_mode` 方式处理。如果 `padding_mode` 设置为 `0`，则使用 :math:`0` 来表示出界的网格位置。
    如果 `padding_mode` 设置为 `border`，对于出界网格位置，则使用border值。如果 `padding_mode` 设置为 `reflection`，请使用边界所反映的位置的值用于指定出界网格位置。对于
    远离边界的位置，它会一直被反射，直到在边界内。

    参数：
        - **input** (Tensor) - 4-D场景下，shape为 :math:`(N, C, H_{in}, W_{in})`，5-D场景下，shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})`。数据类型为float32或float64。
        - **grid** (Tensor) - 4-D场景下，shape为 :math:`(N, H_{out}, W_{out}, 2)`，5-D场景下，shape为 :math:`(N, D_{out}, H_{out}, W_{out}, 3)`。数据类型与 `input` 保持一致。
        - **mode** (str) - 插值方法。可选方法为 `bilinear`，`nearest` 或 `bicubic`。默认值：`bilinear`。需要注意的是 `bicubic` 只支持4-D输入。当 `mode`
          为 `bilinear`，且输入为5-D，则 `mode` 为 `trilinear`。但是，当输入为4-D，则 `mode` 为 `bilinear`。
        - **padding_mode** (str) - 填充方法。可选方法为 `zeros`，`border` 和 `reflection`。默认值为 `zeros`。
        - **align_corners** (bool) - 布尔值。如果设置成 `True`，-1和1被视为引用输入角像素的中心点。如果设置为 `False`，将被视为引用到输入角像素的角点，使采样更不受分辨率影响。
          默认值为 `False`。

    返回：
        Tensor，数据类型与 `input` 相同，4-D场景下，shape为 :math:`(N, C, H_{out}, W_{out})`，5-D场景下，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})`。

    异常：
        - **TypeError** - 如果 `input` 或 `grid` 不是Tensor类型。
        - **TypeError** - 如果 `input` 和 `grid` 的数据类型不一致。
        - **TypeError** - 如果 `input` 或 `grid` 的数据类型无效。
        - **TypeError** - 如果 `align_corners` 不是一个布尔值。
        - **ValueError** - 如果 `input` 或 `grid` 的维度不是四维或五维。
        - **ValueError** - 如果 `input` 的第一个维度不等于 `grid` 的第一个维度。
        - **ValueError** - 如果 `grid` 最后一个维度不等于2（4-D场景）或者3（5-D场景）。
        - **ValueError** - 如果 `mode` 不是 `bilinear`，`nearest`，`bicubic`，数据类型不为String。
        - **ValueError** - 如果 `padding_mode` 不是 `zeros`，`border`，`reflection`，数据类型不为String。
