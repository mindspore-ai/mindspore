mindspore.nn.Flatten
====================

.. py:class:: mindspore.nn.Flatten

    对输入Tensor的第0维之外的维度进行展平操作。

    **输入：**

    - **x** (Tensor) - 要展平的输入Tensor。shape为 :math:`(N, *)`，其中 :math:`*` 表示任意的附加维度。数据类型为 `number <https://www.mindspore.cn/docs/api/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_。

    **输出：**

    Tensor，shape为 :math:`(N, X)`，其中 :math:`X` 是输入 `x` 的shape除N之外的其余维度的乘积。

    **异常：**

    **TypeError** - `x` 不是Tensor。
