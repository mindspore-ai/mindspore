mindspore.nn.probability.bnn_layers.WithBNNLossCell
===================================================

.. py:class:: mindspore.nn.probability.bnn_layers.WithBNNLossCell(backbone, loss_fn, dnn_factor=1, bnn_factor=1)

    为 BNN 生成一个合适的 WithLossCell，用损失函数包装贝叶斯网络。

    参数：
        - **backbone** (Cell) - 目标网络。
        - **loss_fn** (Cell) - 用于计算损失的损失函数。
        - **dnn_factor** (int, float) - backbone 的损失系数，由损失函数计算。默认值：1。
        - **bnn_factor** (int, float) - KL 损失系数，即贝叶斯层的 KL 散度。默认值：1。

    输入：
        - **data** (Tensor) - `data` 的 shape :math:`(N, \ldots)`。
        - **label** (Tensor) - `label` 的 shape :math:`(N, \ldots)`。

    输出：
        Tensor，任意 shape 的标量 Tensor。 

    .. py:method:: backbone_network
        :property:

        返回backbone_network。

        返回：
            Cell，backbone_network。
