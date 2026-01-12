mindspore.ops.swiglu
====================

.. py:function:: mindspore.ops.swiglu(input, dim=-1)

    计算Swish门线性单元函数（Swish Gated Linear Unit function）。
    SwiGLU是 :class:`mindspore.ops.GLU` 激活函数的变体，定义为：

    .. math::
        {SwiGLU}(a, b)= Swish(a) \otimes b

    其中，:math:`a` 表示输入 `input` 拆分后Tensor的前一半元素，:math:`b` 表示输入拆分Tensor的另一半元素，
    Swish(a)=a :math:`\sigma` (a)，:math:`\sigma` 是 :func:`mindspore.ops.sigmoid` 函数， :math:`\otimes` 是Hadamard乘积。

    .. warning::
        只支持 `Atlas A2` 训练系列产品。

    参数：
        - **input** (Tensor) - 被分Tensor，shape为 :math:`(\ast_1, N, \ast_2)` ，其中 `*` 为任意额外维度。 :math:`N` 必须能被2整除。
        - **dim** (int，可选) - 指定分割轴。数据类型为整型，默认 ``-1`` ，输入input的最后一维。

    返回：
        Tensor，数据类型与输入 `input` 相同，shape为 :math:`(\ast_1, M, \ast_2)`，其中 :math:`M=N/2` 。

    异常：
        - **TypeError** - `input` 数据类型不是float16、float32或bfloat16。
        - **TypeError** - `input` 不是Tensor。
        - **RuntimeError** - `dim` 指定维度不能被2整除。

