mindspore.ops.Softmax
======================

.. py:class:: mindspore.ops.Softmax(axis=-1)

    在指定轴上使用Softmax函数做归一化操作。

    更多参考详见 :func:`mindspore.ops.softmax` 。

    参数：
        - **axis** (Union[int, tuple]，可选) - 指定Softmax操作的轴。默认值： ``-1`` 。

    输入：
        - **input** (Tensor) - shape：:math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，数据类型和shape与 `input` 相同。
