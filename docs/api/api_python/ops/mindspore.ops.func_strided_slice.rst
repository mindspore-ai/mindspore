mindspore.ops.strided_slice
===========================

.. py:function:: mindspore.ops.strided_slice(input_x, begin, end, strides, begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0)

    对输入Tensor根据步长和索引进行切片提取。

    该算子在给定的 `input_x` 中提取大小为 `(end-begin)/strides` 的片段。根据起始索引、结束索引和步长进行提取，直到所有维度的索引都不小于结束索引为止, 返回提取出的切片。

    .. note::
        - `begin` 、 `end` 和 `strides` 的shape必须相同。
        - `begin` 、 `end` 和 `strides` 是一维Tensor，且shape小于或等于 `input_x` 的维度。

    切片时，第i维从索引 `begin[i]` 开始取到索引 `end[i]` ，步长为 `strides[i]`。

    例：`input_x` 是shape为 :math:`(5, 6, 7)` 的三维Tensor， `begin` 为(1, 3, 2), `end` 为(3, 5, 6), `strides` 为(1, 1, 2)。切片时，第零维从索引1开始取到3，步长为1；第一维从索引3开始取到5，步长为1；第二维从索引2开始取到6，步长为2。相当于Python式切片 `input_x[1:3, 3:5, 2:6:2]` 。

    如果 `begin` 、 `end` 和 `strides` 的长度小于 `input_x` 的维度， 则缺失的维度默认切取所有的元素， 相当于 `begin` 用0补足， `end` 用相应维度的长度补足， `strides` 用1补足。

    例：`input_x` 是shape为 :math:`(5, 6, 7)` 的三维Tensor， `begin` 为(1, 3), `end` 为(3, 5), `strides` 为(1, 1)。切片时，第零维从索引1开始取到3，步长为1；第一维从索引3开始取到5，步长为1；第二维从索引0开始取到6，步长为1。相当于Python式切片 `input_x[1:3, 3:5, 0:7]` 。

    mask（掩码）的工作原理：

    对于每个特定的mask，内部先将其转化为二进制表示，然后倒序排布后进行计算。比如，对于一个shape为 :math:`(5, 6, 7)` 的Tensor，mask设置为3，3转化为二进制表示为0b011，倒序后为0b110，则该mask只在第零维和第一维产生作用。下面各自举例说明，为简化表达，后面提到的mask都表示转换为二进制并且倒序后的值。

    - `begin_mask` 和 `end_mask`

      如果 `begin_mask` 的第i位为1，则忽略 `begin[i]`，第i维从索引0开始取；若 `end_mask` 的第i位为1，则忽略 `end[i]`，结束的位置为可以取到的最大范围。
      对于shape为 :math:`(5, 6, 7, 8)` 的Tensor，若 `begin_mask` 为0b110，`end_mask` 为0b011，将得到切片 `input_x[0:3, 0:6, 2:7:2]` 。

    - `ellipsis_mask`

      如果 `ellipsis_mask` 的第i位为1，则将在其他维度之间插入所需的任意数量的未指定维度。 `ellipsis_mask` 中只允许一个非零位。
      对于shape为 :math:`(5, 6, 7, 8)` 的Tensor， `input_x[2:,...,:6]` 等同于 `input_x[2:5,:,:,0:6]` ， `input_x[2:,...]` 等同于 `input_x[2:5,:,:,:]` 。

    - `new_axis_mask`

      如果 `new_axis_mask` 的第i位为1，则在输出的第i维添加新的长度为1的维度，并忽略第i维的 `begin` 、 `end` 和 `strides` 。
      对于shape为 :math:`(5, 6, 7)` 的Tensor，若 `new_axis_mask` 为0b010，则第二维将新增一维，输出shape为 :math:`(5, 1, 6, 7)` 的Tensor。

    - `shrink_axis_mask`

      如果 `shrink_axis_mask` 的第i位为1，则第i维被收缩掉，忽略 `begin[i]` 、 `end[i]` 和 `strides[i]` 索引处的值。
      对于shape为 :math:`(5, 6, 7)` 的Tensor，若 `shrink_axis_mask` 为0b010， 则第一维收缩掉，相当于切片 `x[:, 5, :]` 使得输出shape为 :math:`(5, 7)` 。

    .. note::
        `new_axis_mask` 和 `shrink_axis_mask` 不建议同时使用，可能会产生预料之外的结果。

    参数：
        - **input_x** (Tensor) - 需要切片处理的输入Tensor。
        - **begin** (tuple[int]) - 指定开始切片的索引。仅支持大于或等于0的int值。
        - **end** (tuple[int]) - 指定结束切片的索引。仅支持大于0的int值。
        - **strides** (tuple[int]) - 指定各维度切片的步长。输入为一个tuple，仅支持int值。`strides` 的元素必须非零。可能为负值，这会导致反向切片。
        - **begin_mask** (int，可选) - 表示切片的起始索引掩码。默认值：0。
        - **end_mask** (int，可选) - 表示切片的结束索引掩码。默认值：0。
        - **ellipsis_mask** (int，可选) - 维度掩码值为1说明不需要进行切片操作。为int型掩码。默认值：0。
        - **new_axis_mask** (int，可选) - 表示切片的新增维度掩码。默认值：0。
        - **shrink_axis_mask** (int，可选) - 表示切片的收缩维度掩码。为int型掩码。默认值：0。

    返回：
        返回根据起始索引、结束索引和步长进行提取出的切片Tensor。

    异常：
        - **TypeError** - `begin_mask` 、 `end_mask` 、 `ellipsis_mask` 、 `new_axis_mask` 或 `shrink_axis_mask` 不是int。
        - **TypeError** - `begin` 、 `end` 或 `strides` 不是数据类型为int的tuple。
        - **ValueError** - `begin_mask` 、 `end_mask` 、 `ellipsis_mask` 、 `new_axis_mask` 或 `shrink_axis_mask` 小于0。
        - **ValueError** - `begin` 、 `end` 和 `strides` 的shape不同。
