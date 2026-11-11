mindspore.ops.fold
====================

.. py:function:: mindspore.ops.fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)

    将提取出的滑动局部区域块还原成更大的输出Tensor。

    考虑一个batch的输入Tensor，其shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` ，
    其中 :math:`N` 是批次维度，:math:`C \times \prod(\text{kernel_size})` 为每个滑块内值的总数量(一个滑块有 :math:`\prod(\text{kernel_size})` 个空间位置，
    每个位置都包含一个 `C` 通道的向量)，共有 :math:`L` 个这样的滑块：

    .. math::
        L = \prod_d \left\lfloor\frac{\text{output_size}[d] + 2 \times \text{padding}[d] %
            - \text{dilations}[d] \times (\text{kernel_size}[d] - 1) - 1}{\text{strides}[d]} + 1\right\rfloor,

    其中， :math:`d` 遍历所有的空间维度。

    因此，在最后一个维度(列维度)上 `output` 包含特定块内的所有值。

    `dilation`， `padding` 和 `stride` 决定了滑块如何被取出。

    .. warning::
        - 输入是三维Tensor，其shape为 :math:`(N, C \times \prod(\text{kernel_size}), L)` 。
        - 输出是四维Tensor，其shape为 :math:`(N, C, output\_size[0], output\_size[1], ...)` 。

    .. note::
        原 ``ops.col2im`` 和 ``Tensor.col2im`` 入口已移除，可按以下方式迁移到本接口。
        原输入shape为 ``(N, C, K, L)``，其中 ``K`` 为窗口高度与宽度的乘积；
        迁移时须合并中间两维为 ``(N, C * K, L)``，并将关键字 ``padding_value`` 改为 ``padding``。
        保留原 ``output_size`` Tensor 和窗口、膨胀、填充、步长参数，不能仅替换函数名称或依赖新的默认值。
        假设原输入为 ``x``，可写为：

        .. code-block:: python

            n, channels, kernel_area, blocks = x.shape
            columns = x.reshape(n, channels * kernel_area, blocks)
            output = ops.fold(columns, output_size, kernel_size,
                              dilation=dilation, padding=padding_value, stride=stride)

        底层 ``ops.Col2Im`` 原语继续保留；``Tensor.col2im`` 的旧调用也可迁移到上述函数形式。

    参数：
        - **input** (Tensor) - 三维Tensor，支持的数据类型：float16、float32、float64、complex64和complex128。
        - **output_size** (Tensor) - 包含两个int元素的一维Tensor。
        - **kernel_size** (Union[int, tuple[int], list[int]]) - 滑动窗口的大小。如果类型为tuple或者list，则存在两个int元素分别对应kernel的高度与宽度；如果类型为int，则kernel的高度与宽度均为该值。
        - **dilation** (Union[int, tuple[int], list[int]]，可选) - 窗口的空洞卷积的扩充率。如果类型为tuple或者list，则存在两个int元素分别作用于滑窗的高度和宽度；如果类型为int，则这个值作用于滑窗的高度和宽度。默认 ``1`` 。
        - **padding** (Union[int, tuple[int], list[int]]，可选) - 滑窗的隐式零填充量。如果类型为tuple或者list，则存在的两个int元素分别为滑窗的高度和宽度方向的填充量；如果类型为int，则高度和宽度方向的填充量均为这个int值。默认 ``0`` 。
        - **stride** (Union[int, tuple[int], list[int]]，可选) - 空间维度上滑动的步长。如果类型为tuple或者list，则存在的两个int元素分别为滑窗的高度和宽度方向上的步长；如果类型为int，则高度和宽度方向上的步长均为这个int值。默认 ``1`` 。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `kernel_size` 、 `stride` 、 `dilation` 、 `padding` 的数据类型不是int、tuple或者list。
        - **ValueError** - 如果 `kernel_size`、 `dilation`、 `stride` 包含元素的值小于等于0或者元素个数大于 `2` 。
        - **ValueError** - 如果 `padding` 包含元素的值小于零或者元素个数大于 `2` 。
        - **ValueError** - 如果 `input.shape[1] != kernel_size[0] * kernel_size[1]` 。
        - **ValueError** - 如果 `input.shape[2]` 与计算出的滑块数量不匹配。
