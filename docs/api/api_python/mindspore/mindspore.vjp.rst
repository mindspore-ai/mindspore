mindspore.vjp
=================

.. py:function:: mindspore.vjp(fn, inputs, weights=None, has_aux=False)

    计算给定网络的向量雅可比积（vector-jacobian-product, VJP）。

    参数：
        - **fn** (Union[Function, Cell]) - 待求导的函数或网络。以Tensor为入参，返回Tensor或Tensor数组。
        - **inputs** (Union[Tensor, tuple[Tensor], list[Tensor]]) - 输入网络 `fn` 的入参。
        - **weights** (Union[ParameterTuple, Parameter, list[Parameter]]，可选) - 训练网络中需要返回梯度的网络变量。一般可通过 `weights = net.trainable_params()` 获取。默认值： ``None`` 。
        - **has_aux** (bool，可选) - 若 `has_aux` 为 ``True`` ，只有 `fn` 的第一个输出参与 `fn` 的求导，其他输出将直接返回。此时， `fn` 的输出数量必须超过一个。默认值： ``False`` 。

    返回：
        正向输出和计算 vjp 的功能。

        - **net_output** (Union[Tensor, tuple[Tensor]]) - `fn(inputs)` 的输出。特别是当 `has_aux` 设置为 ``True`` 时， `net_output` 是 `fn(inputs)` 的第一个输出。
        - **vjp_fn** (Function) - 用于求解向量雅可比积的函数。接收shape和type与 `net_output` 一致的输入。
        - **aux_value** (Union[Tensor, tuple[Tensor]], 可选) - 若 `has_aux` 为 ``True``，则返回 `aux_value` 。 `aux_value` 是 `fn(inputs)` 的除第一个外的其他输出，且不参与 `fn` 的求导。

    异常：
        - **TypeError** - `inputs` 或 `v` 类型不符合要求。
