mindspore.dataset.config.set_seed
=================================

.. py:function:: mindspore.dataset.config.set_seed(seed)

    设置数据处理中的随机数生成器的种子。

    设置种子能够控制随机生成器的初始状态，起到固定随机数生成结果的目的。

    .. note::
        该接口将同时设置 `random` 、 `numpy.random` 和 `mindspore.dataset` 模块的随机种子为指定值。

    参数：
        - **seed** (int) - 想要设置的种子值。需为非负数。

    异常：
        - **TypeError** - 当 `seed` 不为int类型。
        - **ValueError** - 当 `seed` 为负数。
