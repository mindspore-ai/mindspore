mindspore.set_seed
===================

.. py:function:: mindspore.set_seed(seed)

    设置全局种子。

    .. note::
        - 全局种子可用于 `numpy.random` 、 `mindspore.common.Initializer` 以及 `mindspore.nn.probability.distribution` 。
        - 如果没有设置全局种子，这些包将会各自使用自己的种子， `numpy.random` 和 `mindspore.common.Initializer` 将会随机选择种子值， `mindspore.nn.probability.distribution` 将会使用零作为种子值。
        -  `numpy.random.seed()` 设置的种子仅能被 `numpy.random` 使用，而该API设置的种子也可被 `numpy.random` 使用，因此推荐使用该API设置所有的种子。
        - 在semi_auto_parallel/auto_parallel模式下，使用 `set_seed` 时，同一节点具有相同形状和相同切分策略的权重将被初始化为相同的结果，否则，将被初始化为不同的结果。

    参数：
        - **seed** (int) - 设置的全局种子。

    异常：
        - **ValueError** - 种子值非法（小于0）。
        - **TypeError** - 种子值非整型数。
