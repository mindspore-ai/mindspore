mindspore.ops.LpNorm
=====================

.. py:class:: mindspore.ops.LpNorm(axis, p=2, keep_dims=False, epsilon=1e-12)

    返回输入矩阵或向量的p-范数。

    .. math::
        output = \|input\|_{p}=\left(\sum_{i=1}^{n}\left|input\right|^{p}\right)^{1 / p}

    参数：
        - **axis** (int,list,tuple) - 指定计算范数的维度。
        - **p** (int，可选) - 范数的阶。默认值： ``2`` 。
        - **keep_dims** (bool，可选) - 输出Tensor是否保留原有的维度。默认值： ``False`` 。
        - **epsilon** (float，可选) - 范数下界，当计算的范数小于此值时，用 `epsilon` 替换该结果。默认值： ``1e-12`` 。

    输入：
        - **input** (Tensor) - 输入Tensor，数据类型为float16或float32。

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
