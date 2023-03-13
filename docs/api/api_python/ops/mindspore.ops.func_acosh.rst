mindspore.ops.acosh
====================

.. py:function:: mindspore.ops.acosh(input)

    逐元素计算输入Tensor的反双曲余弦。

    .. math::
        out_i = \cosh^{-1}(input_i)

    .. warning::
        给定一个输入Tensor `input` ，该函数计算每个元素的反双曲余弦。输入范围为[1, inf]。

    参数：
        - **input** (Tensor) - 需要计算反双曲余弦函数的输入Tensor，其秩的范围必须在[0, 7]。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
