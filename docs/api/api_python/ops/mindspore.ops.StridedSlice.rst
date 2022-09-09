mindspore.ops.StridedSlice
===========================

.. py:class:: mindspore.ops.StridedSlice(begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0)

    对输入Tensor根据步长和索引进行切片提取。

    该算子在给定的 `input_tensor` 中提取大小为 `(end-begin)/stride` 的片段。从起始位置开始，根据步长和索引进行提取，直到所有维度的都不小于结束位置为止。

    给定一个 `input_x[m1, m2, ...、mn]` 。 `begin` 、 `end` 和 `strides` 是长度为n的向量。

    在每个掩码字段中（`begin_mask`、`end_mask`、`ellipsis_mask`、`new_axis_mask`、`shrink_axis_mask`），第i位将对应于第i个m。

    对于每个特定的mask，内部先将其转化为二进制表示，然后倒序排布后进行计算。比如说对于一个5*6*7的Tensor，mask设置为3，3转化为二进制表示为ob011，倒序
    后为ob110，则该mask只在第0维和第1维产生作用，下面各自举例说明。

    如果设置了 `begin_mask` 的第i位，则忽略 `begin[i]`，而使用该维度的最大可能取值范围， `begin_mask` 使用方法与之类似。

    对于5*6*7的Tensor， `x[2:,:3,:]` 等同于 `x[2:5,0:3,0:7]` 。

    如果设置了 `ellipsis_mask` 的第i位，则将在其他维度之间插入所需的任意数量的未指定维度。 `ellipsis_mask` 中只允许一个非零位。
    
    对于5*6*7*8的Tensor， `x[2:,...,:6]` 等同于 `x[2:5,:,:,0:6]` 。 `x[2:,...]` 等同于 `x[2:5,:,:,:]` 。

    如果设置了 `new_axis_mask` 的第i位，则忽略 `begin` 、 `end` 和 `strides` ，并在输出Tensor的指定位置添加新的长度为1的维度。

    对于5*6*7的Tensor， `x[:2, newaxis, :6]` 将产生一个shape为 :math:`(2, 1, 6, 7)` 的Tensor。

    如果设置了 `shrink_axis_mask` 的第i位，则第i维被收缩掉，并忽略 `begin[i]` 、 `end[i]` 和 `strides[i]` 索引处的值。

    对于5*6*7的Tensor， `x[:, 5, :]` 相当于将 `shrink_axis_mask` 设置为2, 使得输出shape为:math:`(5, 7)` 。

    .. note::
        步长可能为负值，这会导致反向切片。 `begin` 、 `end` 和 `strides` 的shape必须相同。 `begin` 和 `end` 是零索引。 `strides` 的元素必须非零。

    参数：
        - **begin_mask** (int) - 表示切片的起始索引。使用二进制flag对输入Tensor不同维度进行标志，第i位设置为1则begin[i]参数对应的第i维度设置无效，表示该维度的起始索引从0开始。默认值：0。
        - **end_mask** (int) - 表示切片的结束索引。功能类似begin_mask。使用二进制flag对输入Tensor不同维度进行标志，第i位设置为1则end参数对应的该维度设置无效，表示该维度切分的结束索引到列表最后，即切分到尽可能大的维度。默认值：0。
        - **ellipsis_mask** (int) - 不为0的维度不需要进行切片操作。为int型掩码。默认值：0。
        - **new_axis_mask** (int) - 如果第i位出现1，则begin、end、stride对所有维度参数无效，并在第i位上增加一个大小为1的维度。为int型掩码。默认值：0。
        - **shrink_axis_mask** (int) - 如果第i位设置为1，则意味着第i维度缩小为1。为int型掩码。默认值：0。

    输入：
        - **input_x** (Tensor) - 需要切片处理的输入Tensor。
        - **begin** (tuple[int]) - 指定开始切片的索引。输入为一个tuple，仅支持常量值。
        - **end** (tuple[int]) - 指定结束切片的索引。输入为一个tuple，仅支持常量值。
        - **strides** (tuple[int]) - 指定各维度切片的步长。输入为一个tuple，仅支持常量值。

    输出：
        Tensor。以下内容介绍了输出。

        在第0个维度中，begin为1，end为2，stride为1，因为 :math:`1+1=2\geq2` ，且interval为 :math:`[1,2)` 。因此，在第0个维度中返回具有 :math:`index = 1` 的元素，例如[[3, 3, 3]，[4, 4, 4]]。

        同样在第一个维度中，interval为 :math:`[0,1)` 。根据第0个维度的返回值，返回具有 :math:`index = 0` 的元素，例如[3, 3, 3]。

        同样在第二个维度中，interval为 :math:`[0,3)` 。根据第1个维度的返回值，返回具有 :math:`index = 0,1,2` 的元素，例如[3, 3, 3]。

        最后，输出为[3, 3, 3]。

    异常：
        - **TypeError** - `begin_mask` 、 `end_mask` 、 `ellipsis_mask` 、 `new_ax_mask` 或 `shrink_ax_mask` 不是int。
        - **TypeError** - `begin` 、 `end` 或 `strides` 不是tuple。
        - **ValueError** - `begin_mask` 、 `end_mask` 、 `ellipsis_mask` 、 `new_axis_mask` 或 `shrink_axis_mask` 小于0。