mindspore.nn.Dropout
====================

.. py:class:: mindspore.nn.Dropout(keep_prob=0.5, p=None)

    随机丢弃层。

    Dropout是一种正则化手段，该算子根据丢弃概率 `p` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。在推理过程中，此层返回与 `x` 相同的Tensor。

    论文 `Dropout: A Simple Way to Prevent Neural Networks from Overfitting <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ 中提出了该技术，并证明其能有效地减少过度拟合，防止神经元共适应。更多详细信息，请参见 `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/pdf/1207.0580.pdf>`_ 。

    .. note::
        - 训练过程中每步对同一通道（或神经元）独立进行丢弃。
        - `keep_prob` 参数会在未来版本删除，请使用 `p` 参数代替它。`p` 表示输入Tensor中元素设置成0的概率。

    参数：
        - **keep_prob** (float) - 废弃。输入神经元保留率，数值范围介于(0, 1]之间。例如，`keep_prob` =0.9，删除10%的神经元。默认值：0.5。
        - **p** (Union(float, int, None)) - 输入神经元丢弃率，数值范围介于[0, 1)之间。例如，`p` =0.9，删除90%的神经元。默认值：None。

    输入：
        - **x** (Tensor) - Dropout的输入，任意维度的Tensor。数据类型必须为float16或float32。

    输出：
        Tensor，输出为Tensor，其shape与 `x` shape相同。

    异常：
        - **TypeError** - `keep_prob` 不是浮点数。
        - **TypeError** - `p` 数据类型不是float或int。
        - **TypeError** - `x` 的dtype既不是float16也不是float32。
        - **ValueError** - `keep_prob` 不在范围(0, 1]之间。
        - **ValueError** - `p` 不在范围[0, 1)之间。
        - **ValueError** - `x` 的shape长度小于1。
