mindspore.ops.Softmax
======================

.. py:class:: mindspore.ops.Softmax(axis=-1)

    在指定轴上使用Softmax函数做归一化操作。

    更多参考详见 :func:`mindspore.ops.softmax` 。

    参数：
        - **axis** (Union[int, tuple]) - 指定Softmax操作的轴。默认值： ``-1`` 。

    输入：
        - **logits** (Tensor) - shape：:math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。支持数据类型：

          - Ascend： float16、float32。
          - GPU/CPU： float16、float32、float64。

    输出：
        Tensor，数据类型和shape与 `logits` 相同。
