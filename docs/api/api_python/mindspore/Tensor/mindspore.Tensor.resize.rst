mindspore.Tensor.resize
=======================

.. py:method:: mindspore.Tensor.resize(*new_shape)

    将Tensor改为输入的新shape，并将不足的元素补0。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：参数名 `new_shape` 修改为 `sizes`。

    .. note::
        此方法不更改输入数组的大小，也不返回NumPy中的任何内容，而是返回一个具有输入大小的新Tensor。不支持NumPy参数 `refcheck` 。

    参数：
        - **new_shape** (Union[int, tuple(int)]) - 指定Tensor的新shape。

    返回：
        Tensor。

    .. seealso::
        - :func:`mindspore.Tensor.reshape` ：为Tensor指定新的shape，不改变其数据。
        - :func:`mindspore.Tensor.repeat` ：重复Tensor的元素。
