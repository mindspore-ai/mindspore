mindspore.ops.AddN
===================

.. py:class:: mindspore.ops.AddN(*args, **kwargs)

    按元素将所有输入的Tensor相加。

    所有输入Tensor必须具有相同的shape。

    **输入：**

    - **x** (Union(tuple[Tensor], list[Tensor])) - 输入tuple或list由多个Tensor组成，其数据类型为Number或bool，用于相加。

    **输出：**

    Tensor，与 `x` 的每个Tensor具有相同的shape和数据类型。

    **异常：**

    - **TypeError** - `x` 既不是tuple，也不是list。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> class NetAddN(nn.Cell):
    ...     def __init__(self):
    ...         super(NetAddN, self).__init__()
    ...         self.addN = ops.AddN()
    ...
    ...     def construct(self, *z):
    ...         return self.addN(z)
    ...
    >>> net = NetAddN()
    >>> x = Tensor(np.array([1, 2, 3]), mindspore.float32)
    >>> y = Tensor(np.array([4, 5, 6]), mindspore.float32)
    >>> output = net(x, y, x, y)
    >>> print(output)
    [10. 14. 18.]
