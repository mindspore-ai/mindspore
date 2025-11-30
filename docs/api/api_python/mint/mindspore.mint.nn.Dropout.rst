mindspore.mint.nn.Dropout
=========================

.. py:class:: mindspore.mint.nn.Dropout(p=0.5, inplace=False)

    随机丢弃层。

    Dropout是一种正则化手段，通过阻止神经元节点间的相关性来减少过拟合。在训练过程中，该模块会以丢弃概率 `p` ，随机将一些神经元输出置零。返回值会乘以 :math:`\frac{1}{1-p}` 。在推理过程中，此层返回与输入 `input` 相同的tensor。

    论文 `Dropout: A Simple Way to Prevent Neural Networks from Overfitting <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ 中提出了该技术，并证明其能有效地减少过度拟合，防止神经元共适应。更多详细信息，请参见 `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/pdf/1207.0580.pdf>`_ 。

    .. note::
        - 训练过程中，每步对同一通道（或神经元）独立进行丢弃。
        - `p` 表示输入tensor中元素设置成0的概率。

    参数：
        - **p** (float，可选) - 元素被置零的概率。默认 ``0.5`` 。
        - **inplace** (bool，可选) - 是否启用原地更新功能。默认 ``False`` 。

    输入：
        - **input** (Tensor) - 输入tensor。

    输出：
        Tensor。
