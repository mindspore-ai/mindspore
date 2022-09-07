mindspore.ops.Rint
==================

.. py:class:: mindspore.ops.Rint

    逐元素计算最接近输入数据的整数。

    输入：
        - **input_x** (Tensor) - 待计算的Tensor，数据必须是float16、float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 的数据类型不是float16、float32、float64。
