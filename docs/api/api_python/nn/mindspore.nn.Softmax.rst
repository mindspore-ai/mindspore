mindspore.nn.Softmax
====================

.. py:class:: mindspore.nn.Softmax(axis=-1)

    逐元素计算Softmax激活函数，它是二分类函数 :class:`mindspore.nn.Sigmoid` 在多分类上的推广，目的是将多分类的结果以概率的形式展现出来。

    对输入Tensor在轴 `axis` 上的元素计算其指数函数值，然后归一化到[0, 1]范围，总和为1。

    Softmax定义为：

    .. math::
        \text{softmax}(input_{i}) =  \frac{\exp(input_i)}{\sum_{j=0}^{n-1}\exp(input_j)},

    其中， :math:`input_{i}` 是输入Tensor在轴 `axis` 上的第 :math:`i` 个元素。

    参数：
        - **axis** (int，可选) - 指定Softmax运算的轴axis，假设输入 `input` 的维度为input.ndim，则 `axis` 的范围为 `[-input.ndim, input.ndim)` ，-1表示最后一个维度。默认值： ``-1`` 。

    输入：
        - **input** (Tensor) - 用于计算Softmax函数的Tensor。

    输出：
        Tensor，数据类型和shape与 `input` 相同，取值范围为[0, 1]。

    异常：
        - **TypeError** - `axis` 既不是int也不是tuple。
        - **ValueError** - `axis` 是长度小于1的tuple。
        - **ValueError** - `axis` 是一个tuple，其元素不都在 `[-input.ndim, input.ndim)` 范围内。
