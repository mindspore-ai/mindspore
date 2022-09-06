mindspore.ops.norm
==================

.. py:function:: mindspore.ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)

    返回给定Tensor的矩阵范数或向量范数。

    .. math::
        output = sum(abs(input)**p)**(1/p)

    参数：
        - **input_x** (Tensor) - 输入Tensor。数据类型必须为float16或者float32。
        - **axis** (Union[int, list, tuple]) - 指定要计算范数的输入维度。
        - **p** (int) - 范数的值。默认值：2。 `p` 大于等于0。
        - **keep_dims** (bool) - 输出Tensor是否保留原有的维度。默认值：False。
        - **epsilon** (float) - 用于保持数据稳定性的常量。默认值：1e-12。

    返回：
        Tensor，其数据类型与 `input_x` 相同，其维度信息取决于 `axis` 轴以及参数 `keep_dims` 。例如如果输入的大小为 `(2,3,4)` 轴为 `[0,1]` ，输出的维度为 `(4，)` 。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的数据类型不是float16或者float32。
        - **TypeError** - `axis` 不是int，tuple或者list。
        - **TypeError** - `p` 不是int。
        - **TypeError** - `axis` 是tuple或者list但其元素不是int。
        - **TypeError** - `keep_dims` 不是bool。
        - **TypeError** - `epsilon` 不是float。
        - **ValueError** - `axis` 的元素超出范围 `(-len(input_x.shape), len(input_x.shape))` ，其中 `input_x` 指当前Tensor 。
        - **ValueError** - `axis` 的维度rank大于当前Tensor的维度rank。
