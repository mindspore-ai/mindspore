mindspore.ops.BiasAdd
=====================

.. py:class:: mindspore.ops.BiasAdd(data_format="NCHW")

    返回输入Tensor和偏置Tensor之和。

    将1-D偏置Tensor加到输入Tensor中，相加前会把偏置Tensor广播成与 `input_x` 的shape一致。

    **参数：**

    **data_format** (str) - 输入和输出数据的格式。取值为'NHWC'、'NCHW'或'NCDHW'。默认值：'NCHW'。

    **输入：**

    - **input_x** (Tensor) -输入Tensor。shape可以有2~5个维度。数据类型应为float16或float32。
    - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C)`。`bias` 的shape必须与 `input_x` 的通道维度相同。数据类型应为float16或float32。

    **输出：**

    Tensor，shape和数据类型与 `input_x` 相同。

    **异常：**

    - **TypeError** - `data_format` 不是str。
    - **TypeError** - `input_x` 或  `bias` 不是Tensor。
    - **TypeError** - `input_x` 或  `bias` 的数据类型既不是float16也不是float32。
    - **TypeError** - `input_x` 的维度不在[2, 5]范围内。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> input_x = Tensor(np.arange(6).reshape((2, 3)), mindspore.float32)
    >>> bias = Tensor(np.random.random(3).reshape((3,)), mindspore.float32)
    >>> bias_add = ops.BiasAdd()
    >>> output = bias_add(input_x, bias)
    >>> print(output.shape)
    (2, 3)
    