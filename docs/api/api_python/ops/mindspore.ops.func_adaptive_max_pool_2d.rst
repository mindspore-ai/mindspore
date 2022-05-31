mindspore.ops.adaptive_max_pool2d
=================================

.. py:function:: mindspore.ops.adaptive_max_pool2d(input_x, output_size, return_indices)

    对输入的多维数据进行二维的自适应最大池化运算。

	通过output_size指定输出的高度 :math:`H_{out}` 和宽度 :math:`W_{out}` 

    一般，输入shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` 的Tensor，输出 :math:`(N_{in}, C_{in}, H_{out}, W_{out})` 维上的区域最大值。运算如下：	

    .. math::

        \begin{align}
        h_{start} &= floor(i * H_{in} / H_{out})\\
        h_{end} &= ceil((i + 1) * H_{in} / H_{out})\\
        w_{start} &= floor(j * W_{in} / W_{out})\\
        w_{end} &= ceil((j + 1) * W_{in} / W_{out})\\
        Output(i,j) &= {\max Input[h_{start}:h_{end}, w_{start}:w_{end}]}
        \end{align}

    **参数：**

    - **input_x** (Tensor) - shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, H_{in}, W_{in})` 的tensor，数据类型支持float16, float32, float64。
    - **output_size** (Union[int, tuple]) - 指定输出的高度 :math:`H_{out}` 和宽度 :math:`W_{out}` ， output_size可以是int类型 :math:`H_{out}` ，表示输出的高度和宽度均为 :math:`H_{out}` ；output_size也可以是 :math:`H_{out}` 和 :math:`W_{out}` 组成的tuple类型，其中 :math:`H_{out}` 和 :math:`W_{out}` 为int类型或者None，如果是None，表示与输入相同。
    - **return_indices** (bool) - 如果为True，输出最大值的索引，默认值为False。

    **返回：**

    Tensor，shape为 :math:`(N_{in}, C_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, H_{in}, W_{in})` 的tensor，类型与输入相同。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。
    - **TypeError** - `input_x` 中的数据不是float16, float32, float64.
    - **TypeError** - `output_size` 不是int或者tuple。
    - **TypeError** - `return_indices` 不是bool。
    - **ValueError** - `output_size` 是tuple，但大小不是2。
    - **ValueError** - `input_x` 的维度不是CHW或者NCHW。
