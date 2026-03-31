mindspore.ops.split
====================

.. py:function:: mindspore.ops.split(tensor, split_size_or_sections, axis=0)

    沿指定轴将输入tensor切分成多个子tensor。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`axis` 将改为 `dim`。

    参数：
        - **tensor** (Tensor) - 输入tensor。
        - **split_size_or_sections** (Union[int, tuple(int), list(int)]) - 切分后子tensor的大小。
        - **axis** (int，可选) - 指定轴，默认 ``0`` 。

    .. note::
        - 如果 `split_size_or_sections` 是int类型， `tensor` 将被均匀切分成块，每块大小为 `split_size_or_sections` ，若 `tensor.shape[axis]` 不能被 `split_size_or_sections` 整除，则最后一块大小为余数；
        - 如果 `split_size_or_sections` 是tuple或list类型，`tensor` 将沿 `axis` 轴被切分成 `len(split_size_or_sections)` 块，大小为 `split_size_or_sections` 。

    返回：
        一个由tensor组成的tuple。
