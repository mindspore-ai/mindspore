mindspore.dataset.config.set_fast_recovery
===============================================

.. py:function:: mindspore.dataset.config.set_fast_recovery(fast_recovery)

    在数据集管道故障恢复时，是否开启快速恢复模式（快速恢复模式下，无法保证随机性的数据增强操作得到与故障之前相同的结果）。

    参数：
        - **fast_recovery** (bool) - 是否开启快速恢复模式。

    异常：
        - **TypeError** - `fast_recovery` 不是bool类型。
