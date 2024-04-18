mindspore.ops.stop_gradient
===========================

.. py:function:: mindspore.ops.stop_gradient(value)

    用于消除某个值对梯度的影响，例如截断来自于函数输出的梯度传播。更多细节请参考 `Stop Gradient <https://www.mindspore.cn/tutorials/zh-CN/master/beginner/autograd.html#stop-gradient>`_ 。

    参数：
        - **value** (Any) - 需要被消除梯度影响的值。

    返回：
        一个与 `value` 相同的值。
