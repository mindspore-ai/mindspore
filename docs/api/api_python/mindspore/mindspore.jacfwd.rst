mindspore.jacfwd
====================

.. py:function:: mindspore.jacfwd(fn, grad_position=0, has_aux=False)

    通过前向模式计算给定网络的雅可比矩阵，对应 `前向模式自动微分 <https://www.mindspore.cn/docs/zh-CN/master/design/auto_gradient.html#前向自动微分>`_。当网络输出数量远大于输入数量时，使用前向模式求雅可比矩阵比反向模式性能更好。

    参数：
        - **fn** (Union[Cell, Function]) - 待求导的函数或网络。以Tensor为入参，返回Tensor或Tensor数组。
        - **grad_position** (Union[int, tuple[int]], 可选) - 指定求导输入位置的索引。若为int类型，表示对单个输入求导；若为tuple类型，表示对tuple内索引的位置求导，其中索引从0开始。默认值：0。
        - **has_aux** (bool, 可选) - 若 `has_aux` 为True，只有 `fn` 的第一个输出参与 `fn` 的求导，其他输出将直接返回。此时， `fn` 的输出数量必须超过一个。默认值：False。

    返回：
        Function，用于计算给定函数的雅可比矩阵。例如 `out1, out2 = fn(*args)` ，若 `has_aux` 为True，梯度函数将返回 `(Jacobian, out2)` 形式的结果，其中 `out2` 不参与求导，若为False，将直接返回 `Jacobian` 。

    异常：
        - **TypeError** - `grad_position` 或 `has_aux` 类型不符合要求。
