mindspore.ops.LinSpace
======================

.. py:class:: mindspore.ops.LinSpace

    返回一个在区间 `start` 和 `stop` （包括 `start` 和 `stop` ）内均匀分布的，包含 `num` 个值的1维Tensor。

    .. math::
        \begin{aligned}
        &step = (stop - start)/(num - 1)\\
        &output = [start, start+step, start+2*step, ... , stop]
        \end{aligned}

    **输入：**
    
    - **start** (Tensor) - 0维Tensor，数据类型必须为float32。区间的起始值。
    - **stop** (Tensor) - 0维Tensor，数据类型必须为float32。区间的末尾值。
    - **num** (int) - 间隔中的包含的数值数量，包括区间端点。

    **输出：**
    
    Tensor，与 `start` 的shape和数据类型相同。

    **异常：**
    
    - **TypeError** - `start` 或 `stop` 不是Tensor。
    - **TypeError** - `start` 或 `stop` 的数据类型不是float32。
    - **TypeError** - `num` 不是int类型。
