mindspore.ops.GridSampler3D
===========================

.. py:class:: mindspore.ops.GridSampler3D(interpolation_mode='bilinear', padding_mode='zeros', align_corners=False)

    给定一个输入和一个网格，使用网格中的输入值和像素位置计算输出。只支持体积(5-D)的输入。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.grid_sample`。

    参数：
        - **interpolation_mode** (str，可选) - 指定插值方法。可选方法为 ``'bilinear'`` 、 ``'nearest'`` 或 ``'bicubic'`` 。默认值： ``'bilinear'`` 。
        - **padding_mode** (str，可选) - 指定填充方法。可选方法为 ``'zeros'`` 、 ``'border'`` 和 ``'reflection'`` 。默认值为 ``'zeros'`` 。
        - **align_corners** (bool，可选) - 指定对齐方式。如果设置成 ``True`` ，-1和1被视为引用输入角像素的中心点。如果设置为 ``False`` ，将被视为引用到输入角像素的角点，使采样更不受分辨率影响。
          默认值为 ``False`` 。

    输入：
        - **input_x** (Tensor) - 5-D输入Tensor，shape为 :math:`(N, C, D_{in}, H_{in}, W_{in})`。数据类型为float32或float64。
        - **grid** (Tensor) - 5-D Tensor，shape为 :math:`(N, D_{out}, H_{out}, W_{out}, 3)`。数据类型与 `input_x` 保持一致。

    输出：
        Tensor，数据类型与 `input_x` 相同，shape为 :math:`(N, C, D_{out}, H_{out}, W_{out})`。
