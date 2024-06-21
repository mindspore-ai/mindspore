mindspore.mint.nn.functional.unfold
===================================

.. py:function:: mindspore.mint.nn.functional.unfold(input, kernel_size, dilation=1, padding=0, stride=1)

    从一个batch的输入Tensor中提取滑动局部块。

    考虑一个batch的输入Tensor，其shape为 :math:`(N, C, *)` ，其中 :math:`N` 是batch维度，
    :math:`C` 是channel维度， :math:`*` 表示任意的空间维度。此操作将展平输入 `x` 空间维度内每个滑动的
    `kernel_size` 大小的滑块为输出的3-D Tensor中的一列(如，最后一维)，输出Tensor的shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` ，
    其中 :math:`C \times \prod(\text{kernel_size})` 为每个滑块内值的总数量(一个滑块有 :math:`\prod(\text{kernel_size})` 个空间位置，
    每个位置都包含一个 `C` 通道的向量)，共有 :math:`L` 个这样的滑块：

    .. math::
        L = \prod_d \left\lfloor\frac{\text{spatial_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    其中， :math:`\text{spatial_size}` 由 `input` 的空间维度(上面的 :math:`*` )决定， :math:`d` 遍历所有的空间维度。

    因此，在最后一个维度(列维度)上 `output` 包含特定块内的所有值。

    `dilation`， `padding` 和 `stride` 决定了滑块如何被取出。

    .. warning::
        - 目前，只支持4-D（包含批维度）的类图像Tensor。
        - 对于Ascend，仅Atlas A2以上平台支持。

    参数：
        - **input** (Tensor) - 四维Tensor。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑窗大小。应该是两个int，分别为滑窗的高度和宽度；如果 `kernel_size` 是int，则高度和宽度均为这个int值。是一个必要参数。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率，应该是两个int，分别作用于滑窗的高度和宽度；如果 `dilation` 是int，则这个值作用于滑窗的高度和宽度。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量，应该是两个int，分别作用于滑窗的高度和宽度；如果是 `padding` 是int，则这个值作用于滑窗的高度和宽度。默认值:  ``0`` 。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长，应该是两个int，分别为滑窗的高和宽方向上的步长；如果 `stride` 是int，则高和宽方向上的步长均为这个int值。默认值:  ``1`` 。

    返回：
        Tensor，数据类型与 `input` 相同。

    Shape:
        - **input** - :math:`(N, C, *)`
        - **output** - :math:`(N, C \times \prod(\text{kernel_size}), L)`

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `dilation` 、 `padding` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `kernel_size` 、 `dilation` 、 `stride` 包含元素的值小于等于0或者元素个数不合法。
        - **ValueError** - 如果 `padding` 包含元素的值小于零。
