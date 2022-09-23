mindspore.ops.jet
=================

.. py:function:: mindspore.ops.jet(fn, primals, series)

    计算函数或网络输出对输入的高阶微分。给定待求导函数的原始输入和自定义的1到n阶导数，将返回函数输出对输入的第1到n阶导数。一般情况，建议输入的1阶导数值为全1，更高阶的导数值为全0，这与输入对本身的导数情况是一致的。

    .. note::
        - 若 `primals` 是int型的Tensor，会被转化成float32格式进行计算。

    参数：
        - **fn** (Union[Cell, function]) - 待求导的函数或网络。
        - **primals** (Union[Tensor, tuple[Tensor]]) - `fn` 的输入，单输入的type为Tensor，多输入的type为Tensor组成的tuple。
        - **series** (Union[Tensor, tuple[Tensor]]) - 输入的原始第1到第n阶导数。若为tuple则长度与数据类型应与 `primals` 一致。type与 `primals` 相同，Tensor第一维度i对应输出对输入的第1到第i+1阶导数。

    返回：
        tuple，由 `out_primals` 和 `out_series` 组成。

        - **out_primals** (Union[Tensor, list[Tensor]]) - `fn(primals)` 的结果。
        - **out_series** (Union[Tensor, list[Tensor]]) - `fn` 输出对输入的第1到n阶导数。

    异常：
        - **TypeError** - `primals` 不是Tensor或tuple。
        - **TypeError** - `primals` 和 `series` 的type不一致。
