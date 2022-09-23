mindspore.ops.vjp
=================

.. py:function:: mindspore.ops.vjp(fn, inputs, v)

    计算给定网络的向量雅可比积(vector-jacobian-product, VJP)。VJP对应 `反向模式自动微分 <https://www.mindspore.cn/docs/zh-CN/r1.9/design/auto_gradient.html#反向自动微分>`_。

    .. note::
        此接口未來会变动。

    参数：
        - **fn** (Union[Function, Cell]) - 待求导的函数或网络。以Tensor为入参，返回Tensor或Tensor数组。
        - **inputs** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 输入网络 `fn` 的入参。
        - **v** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 与雅可比矩阵相乘的向量，shape和type与网络的正向计算结果一致。

    返回：
        - **net_output** (Union[Tensor, tuple[Tensor]]) - 输入网络的正向计算结果。
        - **vjp** (Union[NoneType, int, tuple[int]]) - 向量雅可比积的结果。

    异常：
        - **TypeError** - `inputs` 或 `v` 类型不符合要求。
