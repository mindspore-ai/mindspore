mindspore.ops.addn
===================

.. py:function:: mindspore.ops.addn(x)

    逐元素将所有输入的Tensor相加。

    所有输入Tensor必须具有相同的shape。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - Tensor组成的tuble或list。

    返回：
        Tensor，与 `x` 的每个Tensor具有相同的shape和数据类型。

    异常：
        - **TypeError** - `x` 既不是tuple，也不是list。
        - **ValueError** - `x` 中存在shape不同的Tensor。
