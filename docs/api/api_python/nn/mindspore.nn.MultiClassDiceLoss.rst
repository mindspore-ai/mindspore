mindspore.nn.MultiClassDiceLoss
================================

.. py:class:: mindspore.nn.MultiClassDiceLoss(weights=None, ignore_indiex=None, activation='softmax')

    对于多标签问题，可以将标签通过one-hot编码转换为多个二分类标签。每个通道可以看做是一个二分类问题，所以损失可以通过先计算每个类别的二分类的 :class:`mindspore.nn.DiceLoss` 损失，再计算各二分类损失的平均值得到。

    参数：
        - **weights** (Union[Tensor, None]) - Shape为 :math:`(num\_classes, dim)` 的Tensor。权重shape[0]应等于标签shape[1]。默认值： ``None`` 。
        - **ignore_indiex** (Union[int, None]) - 指定需要忽略的类别序号，如果为None，计算所有类别的Dice Loss值。默认值： ``None`` 。
        - **activation** (Union[str, Cell]) - 应用于全连接层输出的激活函数，如'ReLU'。取值范围：[ ``'softmax'`` , ``'logsoftmax'`` , ``'relu'`` , ``'relu6'`` , ``'tanh'`` , ``'Sigmoid'`` ]。默认值： ``'softmax'`` 。

    输入：
        - **logits** (Tensor) - shape为 :math:`(N, C, *)` 的Tensor，其中 :math:`*` 表示任意数量的附加维度。logits维度应大于1。数据类型必须为float16或float32。
        - **labels** (Tensor) - shape为 :math:`(N, C, *)` 的Tensor，与 `logits` 的shape相同。标签维度应大于1。数据类型必须为float16或float32。

    输出：
        Tensor，输出为每个样本采样通过MultiClassDiceLoss函数计算所得。

    异常：
        - **ValueError** - `logits` 与 `labels` 的shape不同。
        - **TypeError** - `logits` 或 `labels` 的类型不是Tensor。
        - **ValueError** - `logits` 或 `labels` 的维度小于2。
        - **ValueError** - `weights` 的shape[0]和 `labels` 的shape[1]不相等。
        - **ValueError** - `weights` 是Tensor，但其维度不是2。
