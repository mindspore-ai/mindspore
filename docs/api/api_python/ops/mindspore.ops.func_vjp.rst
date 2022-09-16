mindspore.ops.vjp
=================

.. py:function:: mindspore.ops.vjp(fn, inputs, has_aux=False)

    计算给定网络的向量雅可比积(vector-jacobian-product, VJP)。VJP对应 `反向模式自动微分 <https://www.mindspore.cn/docs/zh-CN/master/design/auto_gradient.html#反向自动微分>`_。

    参数：
        - **fn** (Union[Function, Cell]) - 待求导的函数或网络。以Tensor为入参，返回Tensor或Tensor数组。
        - **inputs** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 输入网络 `fn` 的入参。
        - **has_aux** (bool) - 若 `has_aux` 为True，只有 `fn` 的第一个输出参与 `fn` 的求导，其他输出将直接返回。此时， `fn` 的输出数量必须超过一个。默认值：False。

    返回：
        - **net_output** (Union[Tensor, tuple[Tensor]]) - 输入网络的正向计算结果。
        - **vjp_fn** (Function) - 用于求解向量雅可比积的函数。接收shape和type与 `net_out` 一致的输入。
        - **aux_value** (Union[Tensor, tuple[Tensor]], 可选) - 若 `has_aux` 为True，才返回 `aux_value` 。`aux_value` 是 `fn(inputs)` 的第一个除外的其他输出，且不参与 `fn` 的求导。

    异常：
        - **TypeError** - `inputs` 或 `v` 类型不符合要求。
