mindspore.ops.glu
=================

.. py:function:: mindspore.ops.glu(x, axis=-1)

    门线性单元函数（Gated Linear Unit function）。

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)


    其中，:math:`a` 表示输入input_x 拆分后 Tensor的前一半元素，:math:`b` 表示输入拆分Tensor的另一半元素。

    参数：
        - **x** (Tensor) - :math:`(\ast_1, N, \ast_2)`  。 `x` 必须在 `axis` 轴能够被平均分成两份。
        - **axis** (int) - 指定分割轴。数据类型为整型，默认值：-1。

    返回：
        Tensor，数据类型与输入 `x` 相同，shape等于 `x` 按照 `axis` 拆分后的一半。

    异常：
        - **TypeError** -  `x` 数据类型不是Number。
        - **TypeError** -  `x` 不是Tensor。
