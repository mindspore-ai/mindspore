mindspore.mint.nn.functional.linear
======================================

.. py:function:: mindspore.mint.nn.functional.linear(input, weight, bias=None)

    对输入 `input` 应用全连接操作。全连接定义为：

    .. math::
        output = input * weight^{T} + bias

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor，shape是 :math:`(*, in\_channels)`，其中 :math:`*` 表示任意的附加维度。
        - **weight** (Tensor) - 输入Tensor的权重，shape是 :math:`(out\_channels, in\_channels)` 或 :math:`(in\_channels)`。
        - **bias** (Tensor，可选) - 添加在输出结果的偏差，shape是 :math:`(out\_channels)` 或 :math:`()`。默认值：``None`` ，偏差为0。

    返回：
        输出结果，shape由 `input` 和 `weight` 的shape决定。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `weight` 不是Tensor。
        - **TypeError** - `bias` 不是Tensor。
