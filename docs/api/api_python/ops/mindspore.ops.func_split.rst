mindspore.ops.split
====================

.. py:function:: mindspore.ops.split(x, split_size_or_sections, axis=0)

    根据指定的轴将输入Tensor切分成块。

    参数：
        - **x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **split_size_or_sections** (Union[int, tuple(int), list(int)]) - 如果 `split_size_or_sections` 是int类型，
          `x` 将被均匀的切分成块，每块的大小为 `split_size_or_sections` ，若 `x.shape[axis]` 不能被 `split_size_or_sections` 整除，最后一块大小将小于 `split_size_or_sections` 。
          如果 `split_size_or_sections` 是个list类型，`x` 将沿 `axis` 轴被切分成 `len(split_size_or_sections)` 块，大小为 `split_size_or_sections` 。
        - **axis** (int) - 指定分割轴。默认值：0。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **ValueError** - 参数 `axis` 超出 :math:`[-x.dim, x.dim)` 范围。
        - **TypeError** - `split_size_or_sections` 中的每个元素不是int类型
        - **TypeError** - `split_size_or_sections` 不是int，tuple(int)或list(int)。
        - **ValueError** - `split_size_or_sections` 的和不等于x.shape[axis]。

