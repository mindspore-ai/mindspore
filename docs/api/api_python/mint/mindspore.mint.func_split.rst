mindspore.mint.split
======================

.. py:function:: mindspore.mint.split(tensor, split_size_or_sections, dim=0)

    根据指定的轴将输入Tensor切分成块.

    参数：
        - **tensor** (Tensor) - 要被切分的Tensor。
        - **split_size_or_sections** (Union[int, tuple(int), list(int)]) - 如果 split_size_or_sections 是int类型， tensor将被均匀的切分成块，每块的大小为 split_size_or_sections ，若 tensor.shape[dim] 不能被 split_size_or_sections 整除，最后一块大小将小于 split_size_or_sections 。 如果 split_size_or_sections 是个list类型，tensor 将沿 dim 轴被切分成 len(split_size_or_sections) 块，大小为 split_size_or_sections 。
        - **dim** (int) - 指定分割轴。默认值： ``0`` 。

    返回：
        元素为Tensor的Tuple。

    异常：
        - **TypeError** - `tensor` 不是Tensor。
        - **TypeError** - `dim` 不是int类型。
        - **ValueError** - `dim` 不在[-tensor.ndim, tensor.ndim)范围中。
        - **TypeError** - `split_size_or_sections` 中的每个元素不是int类型。
        - **TypeError** - `split_size_or_sections` 不是int，tuple(int)或list(int)。
        - **ValueError** - `split_size_or_sections` 的和不等于tensor.shape[dim]。