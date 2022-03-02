mindspore.ops.Pad
==================

.. py:class:: mindspore.ops.Pad(paddings)

    根据参数 `paddings` 对输入进行填充。

    例如， 仅填充输入Tensor的最后一个维度，则填充方式为(padding_left, padding_right)；填充输入Tensor的最后两个维度，
    则填充方式为(padding_left, padding_right, padding_top, padding_bottom)；填充最后3个维度，则填充方式为
    (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)。
 
    .. math::
        \begin{aligned}
            &\text{ input_x_shape} = (N_{1},N_{2},...,N_{n}) \\
            &\begin{aligned}
                \text{output_shape = }(&N_{1}+paddings[0,0]+paddings[0,1], \\
                                 & N_{2}+paddings[1,0]+paddings[1,1], \\
                                 &... , \\
                                 & N_{n}+paddings[n-1,0]+paddings[n-1,1])
            \end{aligned}
        \end{aligned}

    **参数：**

    - **paddings** (tuple) - 填充大小，其shape为(N, 2)，N是输入数据的维度，填充的元素为int类型。对于 `x` 的第 `D` 个维度，paddings[D, 0]表示输入Tensor的第 `D` 维度扩展前的大小，paddings[D, 1]表示在输入Tensor的第 `D` 个维度后面要扩展的大小。

    **输入：**

    - **input_x** (Tensor) - Pad的输入，任意维度的Tensor。

    **输出：**

    填充后的Tensor。

    **异常：**

    - **TypeError** - `paddings` 不是tuple。
    - **TypeError** - `input_x` 不是Tensor。
    - **ValueError** - `paddings` 的shape不是 :math:`(N, 2)` 。
    - **ValueError** - `paddings` 的大小不等于2 * len(input_x)。