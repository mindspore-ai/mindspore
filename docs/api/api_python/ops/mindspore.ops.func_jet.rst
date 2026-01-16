mindspore.ops.jet
=================

.. py:function:: mindspore.ops.jet(fn, primals, series)

    计算函数或网络输出对输入的高阶微分。给定待求导函数的原始输入和自定义的1到n阶导数，将返回函数输出对输入的第1到n阶导数。一般情况，建议输入的1阶导数值为全1，更高阶的导数值为全0，这与输入对本身的导数情况是一致的。

    .. note::
        若 `primals` 是int型的tensor，会被转化成float32格式进行计算。

    参数：
        - **fn** (Union[Cell, function]) - 待求导的函数或网络。
        - **primals** (Union[Tensor, tuple[Tensor]]) - `fn` 的输入。
        - **series** (Union[Tensor, tuple[Tensor]]) - 输入的原始第1到第n阶导数。tensor第零维度索引 `i` 对应输出对输入的第 `i+1` 阶导数。

    返回：
        tuple(out_primals, out_series)

        - **out_primals** (Union[Tensor, list[Tensor]]) - `fn(primals)` 的结果。
        - **out_series** (Union[Tensor, list[Tensor]]) - `fn` 输出对输入的第1到n阶导数。
