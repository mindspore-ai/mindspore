mindspore.nn.probability.bnn_layers.NormalPosterior
===================================================

.. py:class:: mindspore.nn.probability.bnn_layers.NormalPosterior(name, shape, dtype=mstype.float32, loc_mean=0, loc_std=0.1, untransformed_scale_mean=-5, untransformed_scale_std=0.1)

    用可训练的参数构建正态分布。

    参数：
        - **name** (str) - 可训练参数的前置名。
        - **shape** (list, tuple) - 均值和标准差的 shape。
        - **dtype** (:class:`mindspore.dtype`) - 用于定义输出张量的数据类型参数。默认值：mindspore.float32。
        - **loc_mean** (int, float) - 初始化可训练参数的分布均值。默认值：0。
        - **loc_std** (int, float) - 初始化可训练参数的分布标准差。默认值：0.1。
        - **untransformed_scale_mean** (int, float) - 初始化可训练参数的分布均值。默认值：-5。
        - **untransformed_scale_std** (int, float) - 初始化可训练参数的分布标准差。默认值：0.1。 

    返回：
        Cell，一种正态分布结果。
