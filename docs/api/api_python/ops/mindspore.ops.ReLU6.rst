mindspore.ops.ReLU6
====================

.. py:class:: mindspore.ops.ReLU6

    计算输入Tensor的ReLU（矫正线性单元），其上限为6。

    更多参考详见 :func:`mindspore.ops.relu6`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意的附加维度数。数据类型必须为float16、float32。

    输出：
        Tensor，数据类型和shape与 `input_x` 相同。
