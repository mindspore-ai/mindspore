mindspore.nn.DiceLoss
======================

.. py:class:: mindspore.nn.DiceLoss(smooth=1e-5)

    Dice系数是一个集合相似性loss，用于计算两个样本之间的相似性。当分割结果最好时，Dice系数的值为1，当分割结果最差时，Dice系数的值为0。

    Dice系数表示两个对象之间的面积与总面积的比率。
    函数如下：

    .. math::
        dice = 1 - \frac{2 * |pred \bigcap true|}{|pred| + |true| + smooth}

    :math:`pred` 表示 `logits` ，:math:`true` 表示 `labels` 。

    参数：
        - **smooth** (float) - 将添加到分母中，以提高数值稳定性的参数。取值大于0。默认值：1e-5。

    输入：
        - **logits** (Tensor) - 输入预测值，任意维度的Tensor。数据类型必须为float16或float32。
        - **labels** (Tensor) - 输入目标值，任意维度的Tensor，一般与 `logits` 的shape相同。数据类型必须为float16或float32。

    输出：
        Tensor，shape为每样本采样的Dice系数的Tensor。

    异常：
        - **ValueError** - `logits` 的维度与 `labels` 不同。
        - **TypeError** - `logits` 或 `labels` 的类型不是Tensor。


