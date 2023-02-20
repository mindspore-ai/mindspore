mindspore.Tensor.new_ones
==========================

.. py:method:: mindspore.Tensor.new_ones(size, *, dtype=None)

    返回一个大小为 `size` 的Tensor，填充值为1。

    参数：
        - **size** (Union[int, tuple, list]) - 定义输出的shape。

    关键字参数：
        - **dtype** (mindspore.dtype, 可选) - 输出的数据类型。默认值：None，返回的Tensor使用和 `self` 相同的数据类型。

    返回：
        Tensor，shape和dtype由输入定义，填充值为1。

    异常：
        - **TypeError** - 如果 `size` 不是一个int，或int的列表/元组。
