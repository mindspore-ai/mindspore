mindspore.ops.PadV3
====================

.. py:class:: mindspore.ops.PadV3(mode="constant", paddings_contiguous=True)

    根据参数 `mode` 和 `paddings_contiguous` 对输入进行填充。

    参数：
        - **mode** (str，可选) - 填充模式，支持 ``"constant"`` 、 ``"reflect"`` 、 ``"edge"`` 和 ``"circular"`` 。默认值： ``"constant"`` 。
          各种填充模式效果如下：

          - ``"constant``": 使用 `constant_value` 指定的值进行填充。
          - ``"reflect``": 通过反射Tensor边界处的像素值，并将反射值沿着Tensor的边界向外扩展来填充输入Tensor。
          - ``"edge``": 使用Tensor边缘上像素的值填充输入Tensor。
          - ``"circular``": 循环填充。在循环填充模式下，图像的像素从一侧被循环地填充到另一侧。例如，右侧的像素将被替换为左侧的像素，底部的像素将被替换为顶部的像素。

        - **paddings_contiguous** (bool，可选) - 是否连续填充。如果为 ``True`` ， `paddings` 格式为[begin0, end0, begin1, end1, ...]，如果为 ``False`` ，`paddings` 格式为[begin0, begin1, ..., end1, end2, ...]。默认值： ``True`` 。

    输入：
        - **x** (Tensor) - 被填充的Tensor，shape为 :math:`(N, *)` ， :math:`*` 为任意数量的额外维度。
        - **paddings** (Tensor) - 指定在 `x` 的每个维度前后填充的零的数量。是一个1D Tensor，数据类型为int32或int64。
        - **constant_value** (Tensor) - `mode` 为'constant'时的填充值，如果未指定则使用零进行填充。数据类型与一致。其数据类型与 `x` 一致。

    输出：
        填充后的Tensor。

    异常：
        - **TypeError** - `x` 或 `paddings` 不是Tensor。
        - **TypeError** - `padding_contiguous` bool。
        - **ValueError** - `mode` 不是string类型或者不在支持的列表里。
        - **ValueError** - `mode` 是"constant"的同时 `paddings` 元素个数不是偶数。
        - **ValueError** - `mode` 是"constant"的同时 `paddings` 元素个数大于输入维度乘以2。
        - **ValueError** - `mode` 是"edge"、"reflect"或"circular"的时 `paddings` 元素个数不是2、4或6。
        - **ValueError** - `mode` 是"edge"、"reflect"或"circular"， `x` 的维度是3， `paddings` 元素个数不是2。
        - **ValueError** - `mode` 是"edge"、"reflect"或"circular"， `x` 的维度是4， `paddings` 元素个数不是4。
        - **ValueError** - `mode` 是"circular"， `x` 的维度是5， `paddings` 元素个数不是6。
        - **ValueError** - `mode` 是"edge"、"reflect"或"circular"的同时 `x` 的维度小于3。
        - **ValueError** - `mode` 是"edge"或"circular"的时 `x` 的维度大于5。
        - **ValueError** - `mode` 是"reflect"的同时 `x` 的维度大于4。
        - **ValueError** - `mode` 是"reflect"的同时填充值大于对应 `x` 的维度。
        - **ValueError** - 填充之后，输出shape数不大于零。
