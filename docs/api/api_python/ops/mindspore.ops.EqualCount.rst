mindspore.ops.EqualCount
=========================

.. py:class:: mindspore.ops.EqualCount

    计算两个Tensor的相同元素的数量。
    
    两个输入Tensor必须具有相同的数据类型和shape。

    输入：
        - **x** (Tensor) - 第一个输入Tensor。如果确定了 `y` 的数据类型和shape，则 `x` 必须与 `y` 相同，反之亦然。 :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **y** (Tensor) - 第二个输入Tensor。如果确定了 `x` 的数据类型和shape，则 `y` 必须与 `x` 相同，反之亦然。

    输出：
        Tensor，数据类型与输入Tensor相同，shape为 :math:`(1,)` 。

    异常：
        - **TypeError** - 如果 `x` 或 `y` 不是Tensor。
        - **ValueError** - 如果 `x` 与 `y` 的shape不相等。
