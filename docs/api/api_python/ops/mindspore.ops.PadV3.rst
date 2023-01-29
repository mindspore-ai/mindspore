mindspore.ops.PadV3
====================

.. py:class:: mindspore.ops.PadV3(mode="constant", paddings_contiguous=True)

    根据参数 `mode` 和 `paddings_contiguous` 对输入进行填充。

    参数：
        - **mode** (str，可选) - 填充模式，支持"constant" 、"reflect" 和 "edge"。默认值："constant"。
        - **paddings_contiguous** (bool，可选) - 是否连续填充。如果为True， `paddings` 格式为[begin0, end0, begin1, end1, ...]，如果为False，`paddings` 格式为[begin0, begin1, ..., end1, end2, ...]。默认值：True。

    输入：
        - **x** (Tensor) - Pad的输入，任意维度的Tensor。
        - **paddings** (Tensor) - Pad的输入，任意维度的Tensor。
        - **constant_value** (Tensor) - Pad的输入，任意维度的Tensor。

    输出：
        填充后的Tensor。

    异常：
        - **TypeError** - `x` 或 `paddings` 不是Tensor。
        - **TypeError** - `padding_contiguous` bool。
        - **ValueError** - `mode` 不是string类型或者不在支持的列表里。
        - **ValueError** - `mode` 是"constant"的同时 `paddings` 元素个数不是偶数。
        - **ValueError** - `mode` 是"constant"的同时 `paddings` 元素个数大于输入维度乘以2。
        - **ValueError** - `mode` 是"edge"或"reflect"的同时 `paddings` 元素个数不是2、4或6。
        - **ValueError** - `mode` 是"edge"或"reflect"， `x` 的维度是3， `paddings` 元素个数不是2。
        - **ValueError** - `mode` 是"edge"或"reflect"， `x` 的维度是4， `paddings` 元素个数不是4。
        - **ValueError** - `mode` 是"edge"或"reflect"的同时 `x` 的维度小于3。
        - **ValueError** - `mode` 是"edge"的同时 `x` 的维度大于5。
        - **ValueError** - `mode` 是"reflect"的同时 `x` 的维度大于4。
        - **ValueError** - `mode` 是"reflect"的同时填充值大于对应 `x` 的维度。
        - **ValueError** - 填充之后，输出shape数不大于零。
