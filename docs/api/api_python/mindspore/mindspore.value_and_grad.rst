mindspore.value_and_grad
============================

.. py:function:: mindspore.value_and_grad(fn, grad_position=0, weights=None, has_aux=False, return_ids=False, sens_param=False)

    生成求导函数，用于计算给定函数的正向计算结果和梯度。

    函数求导包含以下三种场景：

    1. 对输入求导，此时 `grad_position` 非 ``None``，而 `weights` 是 ``None``；
    2. 对网络变量求导，此时 `grad_position` 是 ``None``，而 `weights` 非 ``None``；
    3. 同时对输入和网络变量求导，此时 `grad_position` 和 `weights` 都非 ``None``。

    参数：
        - **fn** (Union[Cell, Function]) - 待求导的网络或函数。
        - **grad_position** (Union[NoneType, int, tuple[int]]，可选) - 指定输入中需要求导的位置索引。默认值： ``0`` 。

          - 若为int类型，表示对单个输入求导；
          - 若为tuple类型，表示对输入中tuple对应的索引位置求导，其中索引从0开始；
          - 若是 ``None``，表示不对输入求导，这种场景下， `weights` 非None。

        - **weights** (Union[ParameterTuple, Parameter, list[Parameter]]，可选) - 训练网络中需要求导的网络变量。一般可通过 `weights = net.trainable_params()` 获取。默认值： ``None`` 。
        - **has_aux** (bool，可选) - 是否返回辅助参数的标志。若为 ``True`` ， `fn` 输出数量必须超过一个，其中只有 `fn` 第一个输出参与求导，其他输出值将直接返回。默认值： ``False`` 。
        - **return_ids** (bool，可选) - 返回的求导函数中是否包含 `grad_position` 或 `weights` 信息。若为 ``True`` ，返回的求导函数中所有的梯度值gradient将被替换为：[gradient, grad_position]或[gradient, weights]。默认值： ``False`` 。
        - **sens_param** (bool) - 是否在输入中配置灵敏度（关于输出的梯度）。如果 `sens_param` 等于 ``False`` ，自动添加一个 `ones_like(output)` 灵敏度。如果 `sens_param` 等于 ``True`` ，灵敏度（输出的梯度）必须通过位置参数或键值对参数来传递，如果是通过键值对参数传递value，那么key必须为sens。默认值： ``False`` 。

    返回：
        Function，用于计算给定函数梯度的求导函数。例如 `out1, out2 = fn(*args)` ，求导函数将返回 `((out1, out2), gradient)` 形式的结果，若 `has_aux` 为 ``True``，那么 `out2` 不参与求导。
        若 `return_ids` 为 ``True`` ，求导函数返回的 `gradient` 将被替代为[gradient, grad_position]或[gradient, weights]。

    异常：
        - **ValueError** - 入参 `grad_position` 和 `weights` 同时为 ``None``。
        - **TypeError** - 入参类型不符合要求。
