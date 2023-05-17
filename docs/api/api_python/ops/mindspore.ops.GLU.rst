mindspore.ops.GLU
=================

.. py:class:: mindspore.ops.GLU(axis=-1)

    门线性单元函数（Gated Linear Unit function）。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.glu`。

    参数：
        - **axis** (int，可选) - 指定分割轴。是一个在范围[-rank(`x`), rank(`x`))内的整数。默认值： ``-1`` ，输入 `x` 的最后一维。

    输入：
        - **x** (Tensor) - 输入Tensor， `x.shape[axis]` 必须为偶数。

    输出：
        Tensor，数据类型与输入 `x` 相同。
