mindspore.nn.Conv1d
======================

.. py:class:: mindspore.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, pad_mode='same', padding=0, dilation=1, group=1, has_bias=False, weight_init='normal', bias_init='zeros')

    对输入Tensor计算一维卷积。该Tensor的shape通常为 :math:`(N, C_{in}, L_{in})` ，其中 :math:`N` 是batch size， :math:`C_{in}` 是空间维度，:math:`L_{in}` 是序列的长度。
    对于每个batch中的Tensor，其shape为 :math:`(C_{in}, L_{in})` ，公式定义如下：

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{in} - 1} \text{ccor}({\text{weight}(C_{\text{out}_j}, k), \text{X}(N_i, k)})

    其中， :math:`ccor` 为 `cross-correlation <https://en.wikipedia.org/wiki/Cross-correlation>`_ ， :math:`C_{in}` 为输入空间维度， :math:`out_{j}` 对应输出的第 :math:`j` 个空间维度，:math:`j` 的范围在 :math:`[0, C_{out}-1]` 内。
    :math:`\text{weight}(C_{\text{out}_j}, k)` 是shape为 :math:`\text{kernel_size}` 的卷积核切片，其中 :math:`\text{kernel_size}` 是卷积核的宽度。 :math:`\text{bias}` 为偏置参数， :math:`\text{X}` 为输入Tensor。
    完整卷积核的shape为 :math:`(C_{out}, C_{in} / \text{group}, \text{kernel_size})` ，其中 `group` 是在空间维度上分割输入 `x` 的组数。
    详细介绍请参考论文 `Gradient Based Learning Applied to Document Recognition <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_ 。

    .. note::
        在Ascend平台上，目前只支持深度卷积场景下的分组卷积运算。也就是说，当 `group>1` 的场景下，必须要满足 `in\_channels` = `out\_channels` = `group` 的约束条件。

    参数：
        - **in_channels** (int) - Conv1d层输入Tensor的空间维度。
        - **out_channels** (int) - Conv1d层输出Tensor的空间维度。
        - **kernel_size** (int) - 指定一维卷积核的宽度。
        - **stride** (int) - 一维卷积核的移动步长，默认值：1。
        - **pad_mode** (str) - 指定填充模式。可选值为 "same"，"valid"，"pad"。默认值："same"。

          - same：输出的宽度与输入整除 `stride` 后的值相同。若设置该模式，`padding` 的值必须为0。
          - valid：在不填充的前提下返回有效计算所得的输出。不满足计算的多余像素会被丢弃。如果设置此模式，则 `padding` 的值必须为0。
          - pad：对输入进行填充。在输入对两侧填充 `padding` 大小的0。如果设置此模式， `padding` 的值必须大于或等于0。

        - **padding** (int) - 输入两侧填充的数量。值应该要大于等于0，默认值：0。
        - **dilation** (int) - 一维卷积核膨胀尺寸。若 :math:`k > 1` ，则kernel间隔 `k` 个元素进行采样。 `k` 取值范围为[1, L]。默认值：1。
        - **group** (int) - 将过滤器拆分为组， `in_channels` 和 `out_channels` 必须可被 `group` 整除。默认值：1。
        - **has_bias** (bool) - Conv1d层是否添加偏置参数。默认值：False。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。它可以是Tensor，str，Initializer或numbers.Number。当使用str时，可选"TruncatedNormal"，"Normal"，"Uniform"，"HeUniform"和"XavierUniform"分布以及常量"One"和"Zero"分布的值，可接受别名"xavier_uniform"，"he_uniform"，"ones"和"zeros"。上述字符串大小写均可。更多细节请参考Initializer的值。默认值："normal"。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。可以使用的初始化方法与"weight_init"相同。更多细节请参考Initializer的值。默认值："zeros"。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。

    输出：
        Tensor，shape为 :math:`(N, C_{out}, L_{out})` 。

        pad_mode为"same"时：

        .. math::
            L_{out} = \left \lceil{\frac{L_{in}}{\text{stride}}} \right \rceil

        pad_mode为"valid"时：

        .. math::
            L_{out} = \left \lceil{\frac{L_{in} - \text{dilation} \times (\text{kernel_size} - 1) }
            {\text{stride}}} \right \rceil

        pad_mode为"pad"时：

        .. math::
            L_{out} = \left \lfloor{\frac{L_{in} + 2 \times padding - (\text{kernel_size} - 1) \times
            \text{dilation} - 1 }{\text{stride}} + 1} \right \rfloor

    异常：
        - **TypeError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 、 `padding` 或 `dilation` 不是int。
        - **ValueError** - `in_channels` 、 `out_channels` 、 `kernel_size` 、 `stride` 或 `dilation` 小于1。
        - **ValueError** - `padding` 小于0。
        - **ValueError** - `pad_mode` 不是"same"，"valid"或"pad"。
