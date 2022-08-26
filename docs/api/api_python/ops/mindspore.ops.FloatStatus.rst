mindspore.ops.FloatStatus
==========================

.. py:class:: mindspore.ops.FloatStatus

    确定元素是否包含非数字（NaN）、正无穷还是负无穷。0表示正常，1表示溢出。

    输入：
        - **x** (Tensor) - 输入Tensor。数据类型必须为float16或float32。 :math:`(N,*)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，shape为 :math:`(1,)` ，数据类型为 `mindspore.dtype.float32` 。

    异常：
        - **TypeError** - 如果 `x` 的数据类型不是float16，float32或float64。
