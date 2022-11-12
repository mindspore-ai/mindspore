mindspore.ops.PadAndShift
==========================

.. py:class:: mindspore.ops.PadAndShift

    使用-1初始化一个Tensor，然后从 `input_x` 转移一个切片到该Tensor。

    .. note::
        如果在Python中使用，PadAndShift按下面流程得到输出Tensor：

        output = [-1] * cum_sum_arr[-1]

        start = cum_sum_arr[shift_idx]

        end = cum_sum_arr[shift_idx + 1]

        output[start:end] = input_x[:(end-start)]

    输入：
        - **input_x** (Tensor) - 输入Tensor，将被转移到 `output` 。
        - **cum_sum_arr** (Tensor) - `cum_sum_arr` 的最后一个值是输出Tensor的长度， `cum_sum_arr[shift_idx]` 是转移起点， `cum_sum_arr[shift_idx+1]` 是转移终点。
        - **shift_idx** (int) - `cum_sum_arr` 的下标。

    输出：
        - **output** (Tensor) - Tensor，数据类型与 `input` 一致。

    异常：
        - **TypeError** - `input_x` 或者 `cum_sum_arr` 不是Tensor。
        - **TypeError** - `shift_idx` 不是int。
        - **ValueError** - `shift_idx` 的值大于等于 `cum_sum_arr` 的长度。

        