mindspore.ops.aminmax
======================

.. py:function:: mindspore.ops.aminmax(input, *, axis=0, keepdims=False)

    返回tensor在指定轴上的最小值和最大值。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `axis` 将重命名为 `dim`，
        `axis` 的默认值将从 ``0`` 调整为 ``None`` 。

    参数：
        - **input** (Tensor) - 输入tensor。

    关键字参数：
        - **axis** (int，可选) - 指定计算的轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``0`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        由两个tensor组成的tuple(min, max)。
