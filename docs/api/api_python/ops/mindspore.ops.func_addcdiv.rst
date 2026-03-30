mindspore.ops.addcdiv
======================

.. py:function:: mindspore.ops.addcdiv(input, tensor1, tensor2, value=1)

    `tensor2` 对 `tensor1` 逐元素相除，将结果乘以标量值 `value` ，并加到 `input` 。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `value` 将改为关键字参数，原有位置传参需要改写。

        变更后的接口签名为 ``mindspore.ops.addcdiv(input, tensor1, tensor2, *, value=1)`` 。

    .. math::
        y[i] = input[i] + value[i] * (tensor1[i] / tensor2[i])

    参数：
        - **input** (Tensor) - 输入tensor。
        - **tensor1** (Tensor) - 被除数（分子）。
        - **tensor2** (Tensor) - 除数（分母）。
        - **value** (Union[Tensor, number]) - （tensor1 / tensor2）的乘数。默认 ``1`` 。

    返回：
        Tensor
