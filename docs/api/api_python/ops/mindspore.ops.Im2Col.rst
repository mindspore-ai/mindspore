mindspore.ops.Im2Col
====================

.. py:class:: mindspore.ops.Im2Col(ksizes, strides=1, dilations=1, pads=0)

    从一个batch的输入Tensor中提取滑动局部块。
    
    考虑一个batch的输入Tensor，其shape为 :math:`(N, C, *)` ，其中 :math:`N` 是batch维度，
    :math:`C` 是channel维度， :math:`*` 表示任意的空间维度。此操作将展平输入 `x` 空间维度内每个滑动的
    `ksize` 大小的滑块为输出的4-D Tensor中的一列(如，最后一维)，输出Tensor的shape为 :math:`(N, C, \prod(\text{kernel_size}), L)` ，
    其中 :math:`C \times \prod(\text{kernel_size})` 为每个滑块内值的总数量(一个滑块有 :math:`\prod(\text{kernel_size})` 个空间位置，
    每个位置都包含一个 `C` 通道的向量)，共有 :math:`L` 个这样的滑块：

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{pads}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,
    
    其中， :math:`\text{spatial_size}` 由输入 `x` 的空间维度(上面的 :math:`*` )决定， :math:`d` 遍历所有的空间维度。

    因此，在最后一个维度(列维度)上 `output` 包含特定块内的所有值。

    `pads` ， `strides` 和 `dilations` 决定了滑块如何被取出。
    
    .. note::
        目前，只支持4-D Tensor(一个batch的图像Tensor)。

    参数：
        - **ksizes** (Union[int, tuple[int], list[int]]) - 内核的大小，应该是两个整数，分别代表高度和宽度。如果是一个整数，则表示高度等于宽度。必须被指定。
        - **strides** (Union[int, tuple[int], list[int]]，可选) - 窗口的滑动步幅，应该是高度和宽度两个整数。如果只有一个整数，则表示高度等于宽度。默认值：1。
        - **dilations** (Union[int, tuple[int], list[int]]，可选) - 窗口的扩张系数，应该是高度和宽度两个整数。如果只有一个整数，则表示高度等于宽度。默认值：1。
        - **pads** (Union[int, tuple[int], list[int]]，可选) - 窗口的填充，必须是1个或2个整数来指定高宽和宽度方向的填充。默认值：0。
        
          - 如果是1个整数，则 :math:`pad\_height = pad\_width` 。
          - 如果是2个整数，则 :math:`pad\_height = pads[0]`, :math:`pad\_width = pads[1]` 。

    输入：
        - **x** (Tensor) - 输入Tensor，只支持4-D Tensor(1个batch的图像Tensor)。支持所有的实数类型。

    输出：
        Tensor，一个4-D Tensor，与输入 `x` 的数据类型相同。

    异常：
        - **TypeError** - 如果 `ksizes` 的类型不在Union[int, tuple[int], list[int]]内。
        - **TypeError** - 如果 `strides` 的类型不在Union[int, tuple[int], list[int]]内。
        - **TypeError** - 如果 `dilations` 的类型不在Union[int, tuple[int], list[int]]内。
        - **TypeError** - 如果 `pads` 类型不在Union[int, tuple[int], list[int]]内。
        - **ValueError** - 如果 `ksizes` 的值不大于0或其元素数量大于2。
        - **ValueError** - 如果 `strides` 的值不大于0或其元素数量大于2。
        - **ValueError** - 如果 `dilations` 的值不大于0或其元素数量大于2。
        - **ValueError** - 如果 `pads` 的值不大于0。
