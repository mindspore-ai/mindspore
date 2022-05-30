mindspore.ops.SmoothL1Loss
==========================

.. py:class:: mindspore.ops.SmoothL1Loss(beta=1.0)

    计算平滑L1损失，该L1损失函数有稳健性。

    平滑L1损失是一种类似于MSELoss的损失函数，但对异常值相对不敏感。参阅论文 `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_ 。

    给定长度为 :math:`N` 的两个输入 `x` 和 `y` ，平滑L1损失的计算如下：

    .. math::
        L_{i} =
        \begin{cases}
        \frac{0.5 (x_i - y_i)^{2}}{\beta}, & \text{if } |x_i - y_i| < \beta \\
        |x_i - y_i| - 0.5 * \beta, & \text{otherwise. }
        \end{cases}

    其中， :math:`\beta` 代表阈值 `beta` 。 :math:`N` 为batch size。

    .. warning::
        此运算符不对损失值执行"reduce"操作。
        如果需要，请调用其他reduce运算符对损失执行"reduce"操作。

    **参数：**
    
    - **beta** (float) - 控制损失函数在L1Loss和L2Loss间变换的阈值。默认值：1.0。
        
    **输入：**
    
    - **logits** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型支持float16或float32。
    - **labels** (Tensor) - shape： :math:`(N, *)` ，与 `logits` 的shape和数据类型相同。

    **输出：**
    
    Tensor，损失值，与 `logits` 的shape和数据类型相同。

    **异常：**
    
    - **TypeError** - `beta` 不是float类型。
    - **TypeError** - `logits` 或 `labels` 的数据类型非float16或float32。
    - **ValueError** - `beta` 小于或等于0。
    - **ValueError** - `logits` 与 `labels` 的shape不同。
