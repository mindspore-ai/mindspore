mindspore.set_seed
===================

.. py:function:: mindspore.set_seed(seed)

    设置全局种子。

    .. note::
        - 全局种子可用于 `numpy.random` 、 `mindspore.common.Initializer` 以及组合全局种子和算子级种子的随机算子，
          例如 :func:`mindspore.ops.uniform` 和 :func:`mindspore.ops.standard_laplace` 。
        - 如果没有设置全局种子，这些包将会各自使用自己的种子， `numpy.random` 和 `mindspore.common.Initializer` 将会随机选择种子值。
        - 对于组合全局种子和算子级种子的随机算子，未设置全局种子时，全局分量为 ``0``；算子种子为 ``None`` 时也按 ``0`` 处理。
          两者均为零时，种子对 ``(0, 0)`` 表示由内核选择随机种子，并不表示用固定种子零生成可重现序列。
          如果只指定正的算子种子，则全局分量仍为零，算子种子计数器随连续调用推进。
          在支持随机数重现的后端、相同执行条件下，新进程中相同的调用序列可以重现结果。
        - 显式调用 ``set_seed(0)`` 与未设置全局种子不同：对于上述随机算子，零会映射为内部固定的默认图种子，
          而 :func:`mindspore.get_seed` 仍返回 ``0``。使用 ``Generator`` 的算子遵循自身的生成器状态，具体行为请参阅对应接口文档。
        -  `numpy.random.seed()` 设置的种子仅能被 `numpy.random` 使用，而该API设置的种子也可被 `numpy.random` 使用，因此推荐使用该API设置所有的种子。
        - 在semi_auto_parallel/auto_parallel模式下，使用 `set_seed` 时，同一节点具有相同形状和相同切分策略的权重将被初始化为相同的结果，否则，将被初始化为不同的结果。

    参数：
        - **seed** (int) - 设置的全局种子。

    异常：
        - **ValueError** - 种子值非法（小于0）。
        - **TypeError** - 种子值非整型数。
