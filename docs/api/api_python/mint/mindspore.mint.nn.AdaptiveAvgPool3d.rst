mindspore.mint.nn.AdaptiveAvgPool3d
===================================

.. py:class:: mindspore.mint.nn.AdaptiveAvgPool3d(output_size)

    对由多个输入平面组成的输入信号，运用三维自适应平均池化。
    
    即对于输入任何尺寸，指定输出的尺寸均为 :math:`(D, H, W)`。
    输出特征的数量等于输入平面的数量。

    假设输入 `input` 最后三维大小分别为 :math:`(inD, inH, inW)`，则输出的最后三维大小分别为 :math:`(outD, outH, outW)`。运算如下：

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0,outD-1], oh \in [0,outH-1], ow \in [0,outW-1]\\
            output[od,oh,ow] = \\
            \qquad mean(input[istartD:iendD+1,istartH:iendH+1,istartW:iendW+1])\\
            where,\\
            \qquad istartD= \left\lceil \frac{od * inD}{outD} \right\rceil \\
            \qquad iendD=\left\lfloor \frac{(od+1)* inD}{outD} \right\rfloor \\
            \qquad istartH=\left\lceil \frac{oh * inH}{outH} \right\rceil \\
            \qquad iendH=\left\lfloor \frac{(oh+1) * inH}{outH} \right\rfloor \\
            \qquad istartW=\left\lceil \frac{ow * inW}{outW} \right\rceil \\
            \qquad iendW=\left\lfloor \frac{(ow+1) * inW}{outW} \right\rfloor
        \end{array}

    .. warning::
        对于Ascend，仅支持 Atlas A2 训练系列产品。

    参数：
        - **output_size** (Union[int, tuple]) - 目标输出尺寸。可以是tuple :math:`(D, H, W)`，也可以是一个int值 `D` 来表示输出尺寸为 :math:`(D, D, D)`。:math:`D`、:math:`H` 和 :math:`W` 可以是int值或者None，其中None表示输出大小与对应的输入的大小相同。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(N, C, D, H, W)` 或 :math:`(C, D, H, W)`。

    输出：
        Tensor。

