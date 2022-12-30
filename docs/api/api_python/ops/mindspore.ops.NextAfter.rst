mindspore.ops.NextAfter
=======================

.. py:class:: mindspore.ops.NextAfter

    逐元素返回 `x1` 指向 `x2` 的下一个可表示值符点值。

    比如有两个数 :math:`a` ， :math:`b` ，数据类型为float32。并且设float32数据类型的可表示值增量为 :math:`eps` 。如果 :math:`a < b` ，那么 :math:`a` 指向 :math:`b` 的下一个可表示值就是 :math:`a+eps` ， :math:`b` 指向 :math:`a` 的下一个可表示值就是 :math:`b-eps` 。

    .. math::
        out_{i} =  nextafter({x1_{i}, x2_{i}})

    更多详细信息请参见 `A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    输入：
        - **x1** (Tensor) - 支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。
        - **x2** (Tensor) - 支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。

    输出：
        Tensor，shape和数据类型与 `x1` 相同。

    异常：
        - **TypeError** - `x1` 和 `x2` 都不是Tensor。
        - **TypeError** - `x1` 和 `x2` 的数据类型非float16或float32。
        - **TypeError** - `x1` 和 `x2` 数据类型不一致。
        - **ValueError** - `x1` 和 `x2` 的shape不一致。
