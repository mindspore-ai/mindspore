mindspore.from_numpy
=======================

.. py:function:: mindspore.from_numpy(array)

    将NumPy数组转换为张量。

    当数据为非C连续时，数据会被拷贝成C连续数据后再创建张量。否则会通过不复制数据的方式把NumPy数组转换为张量。

    参数：
        - **array** (numpy.ndarray) - 输入数组。

    返回：
        与输入数组具有相同数据类型的Tensor。