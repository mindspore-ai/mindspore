mindspore.ops.unfold
====================

.. py:function:: mindspore.ops.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

    从一个batch的输入Tensor中提取滑动局部块。
    
    考虑一个batch的输入Tensor，其shape为 :math:`(N, C, *)` ，其中 :math:`N` 是batch维度，
    :math:`C` 是channel维度， :math:`*` 表示任意的空间维度。此操作将展平输入 `x` 空间维度内每个滑动的
    `kernel_size` 大小的滑块为输出的3-D Tensor中的一列(如，最后一维)，输出Tensor的shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` ，
    其中 :math:`C \times \prod(\text{kernel_size})` 为每个滑块内值的总数量(一个滑块有 :math:`\prod(\text{kernel_size})` 个空间位置，
    每个位置都包含一个 `C` 通道的向量)，共有 :math:`L` 个这样的滑块：

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{pads}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,
    
    其中， :math:`\text{spatial_size}` 由输入 `x` 的空间维度(上面的 :math:`*` )决定， :math:`d` 遍历所有的空间维度。

    因此，在最后一个维度(列维度)上 `output` 包含特定块内的所有值。

    `dilation`， `padding` 和 `stride` 决定了滑块如何被取出。

    .. warning::
        - 在2.0rc1版本，该算子的输出为四维Tensor，其shape为 :math:`(N, C, \prod(\text{kernel_size}), L)` 。
        - 在之后的版本中，其输出则为三维Tensor，其shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` 。 

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 四维Tensor，支持的数据类型: float16、float32、float64、complex64和complex128。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑窗大小。如果是两个int，则分别为滑窗的高度和宽度；如果是一个int，则高度和宽度均为这个int值。是一个必要参数。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率，如果是两个int，则分别作用于滑窗的高度和宽度；如果是一个int，则这个值作用于化窗的高度和宽度。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量。为单个int或者包含一/二个整数的tuple/list。默认值:  ``0`` 。
            
          - 如果是一个int, 则pad_height = pad_width。
          - 如果是两个int, 则pad_height = padding[0], pad_width = padding[1]。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长，如果是两个int，则分别为滑窗的高和宽方向上的步长；如果是一个int，则高和宽方向上的步长均为这个int值。默认值:  ``1`` 。

    返回：
        Tensor，数据类型与 `input` 相同，其shape如上所述。

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `dilation` 、 `padding` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `kernel_size` 、 `dilation` 、 `stride` 包含元素的值小于等于0或者元素个数大于 `2` 。
        - **ValueError** - 如果 `padding` 包含元素的值小于零。
