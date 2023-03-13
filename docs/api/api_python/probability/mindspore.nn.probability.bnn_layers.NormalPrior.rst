mindspore.nn.probability.bnn_layers.NormalPrior
===============================================

.. py:class:: mindspore.nn.probability.bnn_layers.NormalPrior(dtype=mstype.float32, mean=0, std=0.1)

    初始化均值 0 和标准差 0.1 的正态分布。

    参数：
        - **dtype** (:class:`mindspore.dtype`) - 用于定义输出 Tensor 的数据类型的参数。默认值：mstype.float32。 
        - **mean** (int, float) - 正态分布的平均值。默认值：0。
        - **std** (int, float) - 正态分布的标准差。默认值：0.1。

    返回：
        Cell，一种正态分布的结果。
