mindspore.ops.GridSampler3D
===========================

.. py:class:: mindspore.ops.GridSampler3D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=False)

    给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。只支持体积(5-D)的输入。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.grid_sample`。

    参数：
        - **interpolation_mode** (str，可选) - 指定插值方法。可选方法为 ``'bilinear'`` 、 ``'nearest'`` 或 ``'bicubic'`` 。默认值： ``'bilinear'`` 。

          - ``'nearest'``：最近邻插值。每个输出像素的值为最近的输入像素的值。这种方法简单快速，但可能导致块状或像素化的输出。
          - ``'bilinear'``：双线性插值。每个输出像素是最接近的四个输入像素的加权平均值，使用双线性插值计算。与最近邻插值相比，此方法产生更平滑的结果。

        - **padding_mode** (str，可选) - 指定填充模式的可选字符串。可选值为： ``'zeros'`` 、 ``'border'`` 或者 ``'reflection'`` ，默认值： ``'zeros'`` 。
          当采样grid超出输入Tensor的边界时，各种填充模式效果如下：

          - ``'zeros'`` ：使用零填充输入Tensor。
          - ``'border'`` ：使用Tensor边缘上像素的值填充输入Tensor。
          - ``'reflection'`` ：通过反射Tensor边界处的像素值，并将反射值沿着Tensor的边界向外扩展来填充输入Tensor。

        - **align_corners** (bool，可选) - 指定对齐方式。如果设置成 ``True`` ，-1和1被视为引用输入角像素的中心点。如果设置为 ``False`` ，将被视为引用到输入角像素的角点，使采样更不受分辨率影响。
          默认值为 ``False`` 。

    输入：
        - **input_x** (Tensor) - 5-D输入Tensor，shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})`。数据类型为float16、float32或float64。
        - **grid** (Tensor) - 5-D Tensor，shape为 :math:`(N, D_{out}, H_{out}, W_{out}, 3)`。数据类型与 `input_x` 保持一致。

    输出：
        Tensor，数据类型与 `input_x` 相同，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})`。
