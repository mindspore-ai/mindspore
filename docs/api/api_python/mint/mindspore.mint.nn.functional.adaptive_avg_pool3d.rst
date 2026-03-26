mindspore.mint.nn.functional.adaptive_avg_pool3d
================================================

.. py:function:: mindspore.mint.nn.functional.adaptive_avg_pool3d(input, output_size)

    对一个多平面输入信号执行三维自适应平均池化。对于任何输入尺寸，指定输出的尺寸都为 :math:`(D, H, W)`，但是输入和输出特征的数目不会变化。

    假设输入 `input` 最后三维大小分别为 :math:`(D_{in}, H_{in}, W_{in})`，则输出的最后三维大小分别为 :math:`(D_{out}, H_{out}, W_{out})`，运算如下：

    .. math::
        \begin{array}{ll} \\
            \forall \quad od \in [0, D_{out}-1], oh \in [0, H_{out}-1], ow \in [0, W_{out}-1] \\
            output[od,oh,ow] = \\
            \qquad mean(x[D_{istart}:D_{iend}+1,H_{istart}:H_{iend}+1,W_{istart}:W_{iend}+1]) \\
            where, \\
            \qquad D_{istart}= \left\lceil \frac{od * D_{in}}{D_{out}} \right\rceil \\
            \qquad D_{iend}=\left\lfloor \frac{(od+1) * D_{in}}{D_{out}} \right\rfloor \\
            \qquad H_{istart}=\left\lceil \frac{oh * H_{in}}{H_{out}} \right\rceil \\
            \qquad H_{iend}=\left\lfloor \frac{(oh+1) * H_{in}}{H_{out}} \right\rfloor \\
            \qquad W_{istart}=\left\lceil \frac{ow * W_{in}}{W_{out}} \right\rceil \\
            \qquad W_{iend}=\left\lfloor \frac{(ow+1) * W_{in}}{W_{out}} \right\rfloor
        \end{array}

    参数：
        - **input** (Tensor) - adaptive_avg_pool3d的输入，是4D或者5D的Tensor。
        - **output_size** (Union[int, tuple]) - 指定输出特征图的尺寸，可以是个tuple :math:`(D, H, W)`，也可以是一个int值D来表示输出尺寸为 :math:`(D, D, D)` 。:math:`D` 、 :math:`H` 和 :math:`W` 可以是int值或者 ``None``，其中 ``None`` 表示输出大小与对应的输入的大小相同。

    返回：
        Tensor，与输入 `input` 的数据类型相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 维度不是4D或者5D。
        - **ValueError** - `output_size` 的值不是正数。
