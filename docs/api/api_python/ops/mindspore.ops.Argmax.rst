mindspore.ops.Argmax
=====================

.. py:class:: mindspore.ops.Argmax(axis=-1, output_type=mstype.int32)

    返回输入Tensor在指定轴上的最大值索引。

    更多参考详见 :func:`mindspore.ops.argmax`。

    参数：
        - **axis** (int) - 指定Argmax计算轴。默认值： ``-1`` 。
        - **output_type** (:class:`mindspore.dtype`) - 指定输出数据类型。可选值： ``mstype.int32`` 、 ``mstype.int64`` 。默认值： ``mstype.int32`` 。

    输入：
        - **input_x** (Tensor) - Argmax的输入，shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。

    输出：
        Tensor，输出为指定轴上输入Tensor最大值的索引。
