mindspore.ops.lp_pool1d
=======================

.. py:function:: mindspore.ops.lp_pool1d(x, norm_type, kernel_size, stride=None, ceil_mode=False)

    在输入Tensor上应用1D LP池化运算，可被视为组成一个1D输入平面。

    通常，输入的shape为 :math:`(N, C, L_{in})` 或 :math:`(C, L_{in})`，输出的shape为 :math:`(N, C, L_{out})` 或 :math:`(C, L_{out})`。

    .. math::
        L_{out} = \left\lfloor\frac{L_{in} - \text{kernel_size}}{\text{stride}} + 1\right\rfloor

    公式如下：

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    参数：
        - **x** (Tensor) - shape为 :math:`(N, C, L_{in})` 或 :math:`(C, L_{in})` 的Tensor。
        - **norm_type** (Union[int, float]) - 标准化类型，代表公式里的p，不能为0，

          - 如果 p = 1，得到的结果为池化核内元素之和（与平均池化成比例）；
          - 如果 p = :math:`\infty`，得到的结果为最大池化的结果。

        - **kernel_size** (int) - 池化核的尺寸大小。
        - **stride** (int) - 池化操作的移动步长，数据类型为整型。如果值为None，则使用默认值 `kernel_size`。
        - **ceil_mode** (bool) - 若为True，使用ceil来计算输出shape。若为False，使用floor来计算输出shape。默认值：False。

    返回：
        - **output** - LPPool1d的计算结果，shape为 :math:`(N, C, L_{out})` 或 :math:`(C, L_{out})` 的Tensor，与 输入 `x` 的类型一致。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 不是int。
        - **TypeError** - `ceil_mode` 不是bool。
        - **TypeError** - `norm_type` 不是float也不是int。
        - **ValueError** - `norm_type` 等于0。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `x` 的shape长度不等于2或3。