mindspore.ops.GridSampler2D
===========================

.. py:class:: mindspore.ops.GridSampler2D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=False)

    此操作使用基于流场网格的插值对2D `input_x` 进行采样，该插值通常由 :func:`mindspore.ops.affine_grid` 生成。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.grid_sample`。

    参数：
        - **interpolation_mode** (str，可选) - 指定插值方法的可选字符串。可选值为： ``'bilinear'`` 、 ``'nearest'`` ，默认值： ``'bilinear'`` 。

          - ``'nearest'``：最近邻插值。每个输出像素的值为最近的输入像素的值。这种方法简单快速，但可能导致块状或像素化的输出。
          - ``'bilinear'``：双线性插值。每个输出像素是最接近的四个输入像素的加权平均值，使用双线性插值计算。与最近邻插值相比，此方法产生更平滑的结果。

        - **padding_mode** (str，可选) - 指定填充模式的可选字符串。可选值为： ``'zeros'`` 、 ``'border'`` 或者 ``'reflection'`` ，默认值： ``'zeros'`` 。
          当采样grid超出输入Tensor的边界时，各种填充模式效果如下：

          - ``'zeros'`` ：使用零填充输入Tensor。
          - ``'border'`` ：使用Tensor边缘上像素的值填充输入Tensor。
          - ``'reflection'`` ：通过反射Tensor边界处的像素值，并将反射值沿着Tensor的边界向外扩展来填充输入Tensor。

        - **align_corners** (bool，可选) - 一个可选bool。如果为 ``True`` ，输入和输出Tensor的角像素是对齐的，如果为 ``False`` ，
          则不使用角像素对齐。默认值：``False`` 。

    输入：
        - **input_x** (Tensor) - 一个4D的Tensor，shape为 :math:`(N, C, H_{in}, W_{in})` 。支持数据类型：

          - Ascend： float16、float32。
          - GPU/CPU： float16、float32、float64。

        - **grid** (Tensor) - 一个4D的Tensor，dtype和 `input_x` 相同，shape为 :math:`(N, H_{out}, W_{out}, 2)` ，
          用于指定由输入空间维度归一化的采样像素位置。

    输出：
        一个4-D的Tensor，dtype和 `input_x` 相同，shape为 :math:`(N, C, H_{out}, W_{out})` 。
