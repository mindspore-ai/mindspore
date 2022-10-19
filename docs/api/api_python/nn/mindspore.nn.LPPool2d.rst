mindspore.nn.LPPool2d
======================

.. py:class:: mindspore.nn.LPPool2d(norm_type, kernel_size, stride=None, ceil_mode=False)

    对输入的多维数据进行二维平面上的LP池化运算。

    在一个输入Tensor上应用2D LP pooling，可被视为组成一个2D输入平面。

    通常，输入的shape为 :math:`(N, C, H_{in}, W_{in})`，输出的shape为 :math:`(N, C, H_{in}, W_{in})`，输出与输入的shape一致，公式如下：

    .. math::
        f(X) = \sqrt[p]{\sum_{x \in X} x^{p}}

    参数：
        - **norm_type** (Union[int, float]) - 标准化类型，代表公式里的p，

          - 如果 p = 1，得到的结果为池化核内元素之和（与平均池化成比例）；
          - 如果 p = :math:`\infty`，得到的结果为最大池化的结果。

        - **kernel_size** (Union[int, tuple[int]]) - 池化核尺寸大小。如果为整数，则代表池化核的高和宽。如果为tuple，其值必须包含两个整数值分别表示池化核的高和宽。
        - **stride** (Union[int, tuple[int]]) - 池化操作的移动步长，如果为整数，则代表stride的高和宽。如果为tuple，其值必须包含两个整数值分别表示stride的高和宽。如果值为None，则使用默认值 `kernel_size`。
        - **ceil_mode** (bool) - 若为True，使用ceil模式来计算输出shape。若为False，使用floor模式来计算输出shape。默认值：False。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, H_{in}, W_{in})` 的Tensor。

    输出：
        - **output** - LPPool2d的计算结果，shape为 :math:`(N, C, H_{in}, W_{in})`的Tensor，与 输入 `x` 的类型一致。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 不是int也不是tuple。
        - **TypeError** - `ceil_mode` 不是bool。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `kernel_size` 或 `stride` 是一个长度不为2的tuple。
        - **ValueError** - `x` 的shape长度不等于4。