mindspore.ops.LpNorm
=====================

.. py:class:: mindspore.ops.LpNorm(axis, p=2, keep_dims=False, epsilon=1e-12)

    返回输入Tensor的矩阵范数或向量范数。

    .. math::
        output = sum(abs(input)**p)**(1/p)

    参数：
        - **axis** (int,list,tuple) - 指定计算范数的维度。
        - **p** (int，可选) - 范数的阶。默认值：2。
        - **keep_dims** (Tensor，可选) - 输出Tensor是否保留原有的维度。默认值：False。
        - **epsilon** (Tensor，可选) - 添加到分母上的值，以确保数值稳定性。默认值：1e-12。

    输入：
        - **input** (Tensor) - 输入Tensor。

    输出：
        Tensor，数据类型与 `input` 一致，其shape由 `axis` 决定。如果输入shape为 :math:`(2, 3, 4)` ， `axis` 为 :math:`[0, 1]` ，则输出shape为 :math:`(4,)` 。

    异常：
        - **TypeError** - 若 `input` 不是Tensor。
        - **TypeError** - 若 `input` 的数据类型不是float16或float32。
        - **TypeError** - 若 `p` 不是int。
        - **TypeError** - 若 `axis` 不是int、list或tuple。
        - **TypeError** - 若 `axis` 是list或tuple，但含有非int元素。
        - **TypeError** - 若 `keep_dims` 不是bool。
        - **ValueError** - 若 `axis` 的元素不在 :math:`[-r, r)` 范围内，其中 :math:`r` 为 `input` 的秩。
        - **ValueError** - 若 `axis` shape长度大于 `input` shape长度。
