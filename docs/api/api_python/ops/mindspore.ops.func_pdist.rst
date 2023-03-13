mindspore.ops.pdist
===================

.. py:function:: mindspore.ops.pdist(input, p=2.0)

    计算输入中每对行向量之间的p-范数距离。如果输入Tensor `input` 的shape为 :math:`(N, M)`，那么输出就是一个shape为 :math:`(N * (N - 1) / 2,)` 的Tensor。
    如果输入 `input` 的shape为 :math:`(*B, N, M)`，那么输出就是一个shape为 :math:`(*B, N * (N - 1) / 2)` 的Tensor。

    .. math::
        y[n] = \sqrt[p]{{\mid x_{i} - x_{j} \mid}^p}

    其中 :math:`x_{i}, x_{j}` 是输入中两个不同的行向量。

    参数：
        - **input** (Tensor) - 输入Tensor `input` ，其shape为 :math:`(*B, N, M)`，其中 :math:`*B` 表示批处理大小，可以是多维度。类型：float16，float32或float64。
        - **p** (float) - 范数距离的阶， :math:`p∈[0, ∞)`。默认值：2.0。

    返回：
        Tensor，类型与 `input` 一致。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16，float32，float64。
        - **TypeError** - `p` 不是float。
        - **ValueError** - `p` 是负数。
        - **ValueError** - `input` 的维度小于2。
