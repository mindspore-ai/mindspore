mindspore.nn.L1Loss
=============================

.. py:class:: mindspore.nn.L1Loss(reduction='mean')

    L1Loss用于测量 :math:`x` 和 :math:`y` 元素之间的平均绝对误差（MAE），其中 :math:`x` 是输入Tensor和 :math:`y` 是标签Tensor。
    
    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，则计算 :math:`x` 和 :math:`y` 的unreduced loss（即reduction参数设置为"none"）的公式如下：
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    其中， :math:`N` 为batch size。如果 `reduction` 不是"none"，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    **参数：**
    
    **reduction** (str) - 应用于loss的reduction类型。取值为"mean"，"sum"，或"none"。默认值："mean"。

    **输入：**

    - **logits** (Tensor) - shape为 :math:`(N, *)` 的tensor，其中 :math:`*` 表示任意的附加维度。
    - **labels** (Tensor) - shape为 :math:`(N, *)` 的tensor，在通常情况下与 `logits` 的shape相同。但是如果 `logits` 和 `labels` 的shape不同，需要保证他们之间可以互相广播。
          
    **输出：**

    Tensor，为loss float tensor，如果 `reduction` 为'mean'或'sum'，则shape为零；如果 `reduction` 为'none'，则输出的shape为广播的shape。
        
    **异常：**

    **ValueError** - `reduction` 不为"mean"、"sum"或"none"。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> #用例1：logits.shape = labels.shape = (3,)
    >>> loss = nn.L1Loss()
    >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
    >>> labels = Tensor(np.array([1, 2, 2]), mindspore.float32)
    >>> output = loss(logits, labels)
    >>> print(output)
    0.33333334
    >>> #用例2：logits.shape = (3,), labels.shape = (2, 3)
    >>> loss = nn.L1Loss(reduction='none')
    >>> logits = Tensor(np.array([1, 2, 3]), mindspore.float32)
    >>> labels = Tensor(np.array([[1, 1, 1], [1, 2, 2]]), mindspore.float32)
    >>> output = loss(logits, labels)
    >>> print(output)
    [[0. 1. 2.]
     [0. 0. 1.]]
    