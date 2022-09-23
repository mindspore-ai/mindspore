mindspore.ops.jvp
=================

.. py:function:: mindspore.ops.jvp(fn, inputs, v)

    计算给定网络的雅可比向量积(Jacobian-vector product, JVP)。JVP对应 `前向模式自动微分 <https://www.mindspore.cn/docs/zh-CN/r1.9/design/auto_gradient.html#前向自动微分>`_。

    参数：
        - **fn** (Union[Function, Cell]) - 待求导的函数或网络。以Tensor为入参，返回Tensor或Tensor数组。
        - **inputs** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 输入网络 `fn` 的入参。
        - **v** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 与雅可比矩阵相乘的向量，shape和type与网络的输入一致。

    返回：
        - **net_output** (Union[Tensor, tuple[Tensor]]) - 输入网络的正向计算结果。
        - **jvp** (Union[Tensor, tuple[Tensor]]) - 雅可比向量积的结果。

    异常：
        - **TypeError** - `inputs` 或 `v` 类型不符合要求。
