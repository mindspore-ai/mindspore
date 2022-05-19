mindspore.ops.pdist
==================

.. py:function:: mindspore.ops.pdist(x, p)

    计算输入中每对行向量之间的p-范数距离。如果输入`x`的shape为 :math:`(N, M)`，那么输出就是一个shape为 :math:`(N * (N - 1) / 2,)`
    的Tensor。如果`x`的shape为 :math:`(*B, N, M)`，那么输出就是一个shape为 :math:`(*B, N * (N - 1) / 2)`的Tensor。

    .. math::
        y[n] = \sqrt[p]{{\mid x_{i} - x_{j} \mid}^p}

    **参数：**

    - **x** (tensor) - 输入tensor x，其shape为 :math:`(*B, N, M)`，其中 :math:`*B`表示批处理大小，可以是多维度。类型：float16，float32或float64。
    - **p** (float) - P -范数距离的P值，P∈[0，∞]。默认值:2.0。

    **返回：**

    Tensor，类型与 `x` 一致。

    **异常：**

    - **TypeError** - `x` 不是tensor。
    - **TypeError** - `x` 的数据类型不是float16，float32，float64。
    - **TypeError** - `p` 不是float。
    - **ValueError** - `p` 是负数。
    - **ValueError** - `x` 的维度小于2。
    **支持平台：**
    ``CPU``

