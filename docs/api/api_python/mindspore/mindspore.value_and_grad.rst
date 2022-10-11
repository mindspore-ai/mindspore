mindspore.value_and_grad
============================

.. py:function:: mindspore.value_and_grad(fn, grad_position=0, weights=None, has_aux=False)

    生成求导函数，用于计算给定函数的正向计算结果和梯度。

    函数求导包含以下三种场景：

    1. 对输入求导，此时 `grad_position` 非None，而 `weights` 是None;
    2. 对网络变量求导，此时 `grad_position` 是None，而 `weights` 非None;
    3. 同时对输入和网络变量求导，此时 `grad_position` 和 `weights` 都非None。

    参数：
        - **fn** (Union[Cell, Function]) - 待求导的函数或网络。
        - **grad_position** (Union[NoneType, int, tuple[int]]) - 指定求导输入位置的索引。若为int类型，表示对单个输入求导；若为tuple类型，表示对tuple内索引的位置求导，其中索引从0开始；若是None，表示不对输入求导，这种场景下， `weights` 非None。默认值：0。
        - **weights** (Union[ParameterTuple, Parameter, list[Parameter]]) - 训练网络中需要返回梯度的网络变量。一般可通过 `weights = net.trainable_params()` 获取。默认值：None。
        - **has_aux** (bool) - 是否返回辅助参数的标志。若为True， `fn` 输出数量必须超过一个，其中只有 `fn` 第一个输出参与求导，其他输出值将直接返回。默认值：False。

    返回：
        Function，用于计算给定函数的梯度的求导函数。例如 `out1, out2 = fn(*args)` ，梯度函数将返回 `((out1, out2), gradient)` 形式的结果, 若 `has_aux` 为True，那么 `out2` 不参与求导。

    异常：
        - **ValueError** - 入参 `grad_position` 和 `weights` 同时为None。
        - **TypeError** - 入参类型不符合要求。
