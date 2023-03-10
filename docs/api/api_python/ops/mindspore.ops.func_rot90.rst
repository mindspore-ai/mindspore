mindspore.ops.rot90
=======================

.. py:function:: mindspore.ops.rot90(input, k, dims)

    沿轴指定的平面内将n-D Tensor旋转90度。
    如果k>0，旋转方向是从第一轴朝向第二轴，如果k<0，旋转方向从第二轴朝向第一轴。

    参数：
        - **input** (Tensor) - 所输入的Tensor。
        - **k** (int) - 旋转的次数。默认值：1。
        - **dims** (Union[list(int), tuple(int)]) - 要旋转的轴。默认值：[0,1]。

    返回：
        Tensor。

    异常：
        - **TypeError** - 输入不是Tensor。
        - **TypeError** - `k` 不是整数。
        - **TypeError** - `dims` 不是整数组成的list或者tuple。
        - **ValueError** - `dims` 长度不为2。
        - **ValueError** - `dims` 中的元素不在输入Tensor的[-input.ndim, input.ndim)之间。
        - **RuntimeError** - `dims` 的两个元素相同。