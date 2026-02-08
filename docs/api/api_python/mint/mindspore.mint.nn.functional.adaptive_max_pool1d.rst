mindspore.mint.nn.functional.adaptive_max_pool1d
=================================================

.. py:function:: mindspore.mint.nn.functional.adaptive_max_pool1d(input, output_size, return_indices=False)

    对一个多平面输入信号执行一维自适应最大池化。
    也就是说，对于输入任何尺寸，指定输出的尺寸都为L。但是输入和输出特征的数目不会变化。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        Atlas训练系列产品不支持反向传播。

    参数：
        - **input** (Tensor) - adaptive_max_pool1d的输入，为二维或三维的Tensor，数据类型为float16、float32或者float64。。
        - **output_size** (int) - 输出特征图的size。 `output_size` 是一个整数。
        - **return_indices** (bool，可选) - 如果为 ``True`` ，输出最大值的索引，默认值为 ``False`` 。

    返回：
        Union(Tensor, tuple(Tensor, Tensor))。

        - 如果 `return_indices` 为 ``False`` ，返回Tensor, 其shape为 :math:`(N, C_{in}, L_{out})`，数据类型与 `input` 相同。
        - 如果 `return_indices` 为 ``True``，则是一个包含了两个Tensor的Tuple，表示计算结果以及生成max值的位置。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 中的数据不是float16、float32、float64。
        - **TypeError** - `output_size` 不是int或者tuple。
        - **TypeError** - `return_indices` 不是bool。
        - **ValueError** - `output_size` 是tuple，但大小不是1。
