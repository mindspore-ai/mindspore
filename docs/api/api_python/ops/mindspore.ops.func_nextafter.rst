mindspore.ops.nextafter
=======================

.. py:function:: mindspore.ops.nextafter(input, other)

    逐元素返回 `input` 指向 `other` 的下一个可表示值符点值。

    比如有两个数 :math:`a` ， :math:`b` ，数据类型为float32。并且设float32数据类型的可表示值增量为 :math:`eps` 。如果 :math:`a < b` ，那么 :math:`a` 指向 :math:`b` 的下一个可表示值就是 :math:`a+eps` ， :math:`b` 指向 :math:`a` 的下一个可表示值就是 :math:`b-eps` 。

    .. math::
        out_{i} =  nextafter({input_{i}, other_{i}})

    更多详细信息请参见 `A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    参数：
        - **input** (Tensor) - 第一个输入Tensor，支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。
        - **other** (Tensor) - 第二个输入Tensor，支持数据类型为float32和float64。其shape为 :math:`(N,*)` ，其中 :math:`*` 为任意数量的额外维度。

    返回：
        Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 和 `other` 都不是Tensor。
        - **TypeError** - `input` 和 `other` 的数据类型非float16或float32。
        - **TypeError** - `input` 和 `other` 数据类型不一致。
        - **ValueError** - `input` 和 `other` 的shape不一致。