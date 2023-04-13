mindspore.ops.ReLU
==================

.. py:class:: mindspore.ops.ReLU

    线性修正单元激活函数（Rectified Linear Unit）。

    更多参考详见 :func:`mindspore.ops.relu`。

    输入：
        - **input_x** (Tensor) - 输入Tensor，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，
          其数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.html#mindspore.dtype>`_。

    输出：
        shape为 :math:`(N, *)` 的Tensor，数据类型和shape与 `input_x` 相同。
