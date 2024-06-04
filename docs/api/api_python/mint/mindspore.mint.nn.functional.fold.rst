mindspore.mint.nn.functional.fold
=================================

.. py:function:: mindspore.mint.nn.functional.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

    将提取出的滑动局部区域块还原成更大的输出Tensor。

    考虑一个batch的输入Tensor，其shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` ，
    其中 :math:`N` 是批次维度，:math:`C \times \prod(\text{kernel_size})` 为每个滑块内值的总数量(一个滑块有 :math:`\prod(\text{kernel_size})` 个空间位置，
    每个位置都包含一个 `C` 通道的向量)，共有 :math:`L` 个这样的滑块：

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilation}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{stride}[d]} + 1\right\rfloor,

    其中, :math:`d` 遍历所有的空间维度。

    因此，在最后一个维度(列维度)上 `output` 包含特定块内的所有值。

    `dilation`， `padding` 和 `stride` 决定了滑块如何被取出。˝

    .. warning::
        - 目前，只支持3-D或4-D（包含批维度）的类图像输出Tensor。

    参数：
        - **input** (Tensor) - 二维或三维Tensor。
        - **output_size** (Union[int, tuple[int], list[int]]) - 表示输出Tensor的空间维度（如output.shape[2:]）。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑动窗口的大小。如果类型为tuple或者list，则存在两个int元素分别对应kernel的高度与宽度；如果类型为int，则kernel的高度与宽度均为该值。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率。如果类型为tuple或者list，则存在两个int元素分别作用于滑窗的高度和宽度；如果类型为int，则这个值作用于滑窗的高度和宽度。默认值： ``1`` 。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量。如果类型为tuple或者list，则存在的两个int元素分别为滑窗的高度和宽度方向的填充量；如果类型为int，则高度和宽度方向的填充量均为这个int值。默认值： ``0`` 。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长。如果类型为tuple或者list，则存在的两个int元素分别为滑窗的高度和宽度方向上的步长；如果类型为，则高度和宽度方向上的步长均为这个int值。默认值： ``1`` 。

    返回：
        Tensor，数据类型与 `input` 相同。

    Shape:
        - **input** -  :math:`(N, C \times \prod(\text{kernel_size}), L)` 或 :math:`(C \times \prod(\text{kernel_size}), L)`
        - **output** - :math:`(N, C, output\_size[0], output\_size[1], ...)` 或 :math:`(C, output\_size[0], output\_size[1], ...)`

    异常：
        - **TypeError** - 如果 `output_size`、 `kernel_size` 、 `stride` 、 `dilation` 、 `padding` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `output_size` 、 `kernel_size`, `dilation`, `stride` 包含元素的值小于等于0或者元素个数不合法。
        - **ValueError** - 如果 `padding` 包含元素的值小于零或者元素个数不合法。
        - **ValueError** - 如果 `input.shape[-2]` 无法被 `kernel_size` 的乘积整除。
        - **ValueError** - 如果 `input.shape[-1]` 不等于计算出的滑块数量 `L` 。
