mindspore.ops.pad
=================

.. py:function:: mindspore.ops.pad(input_x, paddings)

    根据参数 `paddings` 对输入进行填充。

    输出Tensor的shape计算公式如下：

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

    参数：
        - **input_x** (Tensor) - 输入Tensor。
        - **paddings** (tuple) - 填充大小，其shape为(N, 2)，N是输入数据的维度，填充的元素为int类型。对于 `x` 的第 `D` 个维度，paddings[D, 0]表示输入Tensor的第 `D` 维度前面要扩展（如果该值大于0）或裁剪（如果该值小于0）的大小，paddings[D, 1]表示在输入Tensor的第 `D` 个维度后面要扩展（如果该值大于0）或裁剪（如果该值小于0）的大小。

    返回：
        填充后的Tensor。

    异常：
        - **TypeError** - `paddings` 不是tuple。
        - **TypeError** - `input_x` 不是Tensor。
        - **ValueError** - `paddings` 的shape不是 :math:`(N, 2)` 。
        - **ValueError** - `paddings` 的大小不等于2 * len(input_x)。
        - **ValueError** - 计算出来的输出Tensor的shape里存在0或负数。
