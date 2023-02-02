mindspore.ops.hstack
====================

.. py:function:: mindspore.ops.hstack(x)

    将多个Tensor沿着水平方向进行堆叠。
    对于1-D Tensor，沿第一个轴进行堆叠。其他维度的Tensor沿第二个轴进行堆叠。

    参数：
        - **x** (Union[Tensor, tuple, list]) - 多个1-D 或 2-D的Tensor。对于2-D Tensor，除了第二个轴外，所有的
          Tensor必须有相同的shape。对于1-D Tensor，可拥有任意的长度。

    返回：
        堆叠后的Tensor。

    异常：
        - **TypeError** - 如果 `x` 不是 Tensor、list或tuple。
        - **ValueError** - 如果 `x` 为空。
