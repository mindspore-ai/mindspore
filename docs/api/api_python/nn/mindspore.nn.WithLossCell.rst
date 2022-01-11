mindspore.nn.WithLossCell
=========================

.. py:class:: mindspore.nn.WithLossCell(backbone, loss_fn)

    包含损失函数的Cell。

    封装 `backbone` 和 `loss_fn` 。此Cell接受数据和标签作为输入，并将返回损失函数作为计算结果。

    **参数：**

    - **backbone** (Cell) - 要封装的骨干网络。
    - **loss_fn** (Cell) - 用于计算损失函数。

    **输入：**

    - **data** （Tensor） - shape为 :math:`(N, \ldots)` 的Tensor。
    - **label** （Tensor） - shape为 :math:`(N, \ldots)` 的Tensor。

    **输出：**

    Tensor，loss值，其shape通常为 :math:`()` 。

    **异常：**

    **TypeError**：`data` 或 `label` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
    >>> net_with_criterion = nn.WithLossCell(net, loss_fn)
    >>>
    >>> batch_size = 2
    >>> data = Tensor(np.ones([batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
    >>> label = Tensor(np.ones([batch_size, 10]).astype(np.float32))
    >>>
    >>> output_data = net_with_criterion(data, label)


.. py:method:: backbone_network
    :property:

    获取骨干网络。

    **返回：**

    Cell，骨干网络。
