mindspore.ops.bidense
=====================

.. py:function:: mindspore.ops.bidense(input1, input2, weight, bias=None)

    对输入 `input1` 和 `input2` 应用双线性全连接操作。双线性全连接函数定义如下，

    .. math::
        output = x_{1}^{T}Ax_{2} + b

    其中， :math:`x_{1}` 代表 `input1` ， :math:`x_{2}` 代表 `input2` ， :math:`A` 代表 `weight` ， :math:`b` 代表 `bias` 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input1** (Tensor) - 输入Tensor，shape是 :math:`(*, in1\_channels)` ，其中 :math:`*` 表示任意的附加维度，除最后一维外的维度与 `input2` 保持一致。
        - **input2** (Tensor) - 输入Tensor，shape是 :math:`(*, in2\_channels)` ，其中 :math:`*` 表示任意的附加维度，除最后一维外的维度与 `input1` 保持一致。
        - **weight** (Tensor) - 输入Tensor的权重，shape是 :math:`(out\_channels, in1\_channels, in2\_channels)` 。
        - **bias** (Tensor，可选) - 添加在输出结果的偏差，shape是 :math:`(out\_channels)` 或 :math:`()` 。默认值：``None`` ，偏差为0。

    返回：
        Tensor，shape是 :math:`(*, out\_channels)` ，其中 :math:`*` 表示任意的附加维度。输出Tensor除最后一维外其他维度与所有输入Tensor保持一致。

    异常：
        - **TypeError** - `input1` 不是Tensor。
        - **TypeError** - `input2` 不是Tensor。
        - **TypeError** - `weight` 不是Tensor。
        - **TypeError** - `bias` 不是Tensor。
        - **ValueError** - 如果除了最后一维，`input1` 其他维度与 `input2` 有不同。
