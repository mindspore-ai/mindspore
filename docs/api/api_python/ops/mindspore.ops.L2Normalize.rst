mindspore.ops.L2Normalize
==========================

.. py:class:: mindspore.ops.L2Normalize(axis=0, epsilon=1e-4)

    L2范数归一化算子。

    该算子将对输入 `x` 在给定 `axis` 上的元素进行归一化。函数定义如下：

    .. math::
        \displaylines{{\text{output} = \frac{x}{\sqrt{\text{max}( \sum_{i}^{}\left | x_i  \right | ^2, \epsilon)}}}}

    其中 :math:`\epsilon` 表示 `epsilon` ， :math:`\sum_{i}^{}\left | x_i  \right | ^2` 表示计算输入 `x` 在给定 `axis` 上元素的平方和。

    .. note::
        在Ascend上，暂时不支持float64数据类型。

    参数：
        - **axis** (Union[list(int), tuple(int), int]，可选) - 指定计算L2范数的轴。默认值： ``0`` 。
        - **epsilon** (float，可选) - 为了数值稳定性而引入的很小的浮点数。默认值： ``1e-4`` 。

    输入：
        - **x** (Tensor) - 计算归一化的输入。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意的附加维度数。数据类型必须为float16、float32和float64。

    输出：
        Tensor，shape和数据类型与 `x` 的相同。

    异常：
        - **TypeError** - `axis` 不是list、tuple或int。
        - **TypeError** - `epsilon` 不是float。
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
        - **ValueError** - `x` 的维度不大于0。
