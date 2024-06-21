mindspore.mint.nn.functional.max_pool2d
========================================

.. py:function:: mindspore.mint.nn.functional.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, *, ceil_mode=False, return_indices=False)

    二维最大值池化。

    输入是shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` 的Tensor，输出 :math:`(H_{in}, W_{in})` 维度中的最大值。给定 `kernel_size`
    :math:`ks = (h_{ker}, w_{ker})`，和 `stride` :math:`s = (s_0, s_1)`，运算如下：

    .. math::
        \text{output}(N_i, C_j, h, w) =
        \max_{m=0, \ldots, h_{ker}-1} \max_{n=0, \ldots, w_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times h + m, s_1 \times w + n)

    .. warning::
        只支持 `Atlas A2` 训练系列产品。

    参数：
        - **input** (Tensor) - shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` 的Tensor。在Ascend上，数据类型仅支持float32。
        - **kernel_size** (Union[int, tuple[int]]) - 池化核尺寸大小。可以是一个整数表示池化核的高度和宽度，或者包含两个整数的tuple，分别表示池化核的高度和宽度。
        - **stride** (Union[int, tuple[int], None]) - 池化操作的移动步长。可以是一个整数表示在高度和宽度方向的移动步长，或者包含两个整数的tuple，分别表示在高度和宽度方向的移动步长。默认值： ``None`` ，表示移动步长为 `kernel_size` 。
        - **padding** (Union[int, tuple[int]]) - 池化填充长度。可以是一个整数表示在高度和宽度方向的填充长度，或者包含两个整数的tuple，分别表示在高度和宽度方向的填充长度。默认为 ``0``。
        - **dilation** (Union[int, tuple[int]]) - 控制池化核内元素的间距。默认为 ``1``。
        - **ceil_mode** (bool) - 是否是用ceil代替floor来计算输出的shape。默认为 ``False``。
        - **return_indices** (bool) - 是否输出最大值的索引。默认为 ``False``。

    返回：
        当 `return_indices` 是 ``False`` 时，输出单个 `output` 张量，否则输出一个包含 `output` 张量和 `argmax` 张量的元组。

        - **output** (Tensor) - 输出的池化后的最大值，shape为 :math:`(N_{out}, C_{out}, H_{out}, W_{out})` 。其数据类型与 `input` 相同。

          .. math::
              H_{out} = \left\lfloor\frac{H_{in} + 2 * \text{padding[0]} - \text{dilation[0]} \times (\text{kernel_size[0]} - 1) - 1}{\text{stride[0]}} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in} + 2 * \text{padding[1]} - \text{dilation[1]} \times (\text{kernel_size[1]} - 1) - 1}{\text{stride[1]}} + 1\right\rfloor

        - **argmax** (Tensor) - 输出的最大值对应的索引，在Ascend上，数据类型为int32。仅当 `return_indices` 为True的时候才返回该值。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的维度不是4D。
        - **TypeError** - `kernel_size` 、`stride` 、`padding` 、`dilation` 不是int或者tuple。
        - **ValueError** - `kernel_size`、`stride` 或者 `dilation` 的元素值小于1。
        - **ValueError** - `dilation` 不是全为1。
        - **ValueError** - `padding` 的元素值小于0。
        - **ValueError** - `padding` 的元素值大于 `kernel_size` 的一半。
        - **TypeError** - `ceil_mode` 不是bool类型。
