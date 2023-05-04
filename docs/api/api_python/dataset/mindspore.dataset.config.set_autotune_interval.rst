mindspore.dataset.config.set_autotune_interval
===============================================

.. py:function:: mindspore.dataset.config.set_autotune_interval(interval)

    设置自动数据加速的配置调整step间隔。

    默认设置为 ``0`` ，将在每个epoch结束后调整配置；否则，将每隔 `interval` 个step调整一次配置。

    参数：
        - **interval** (int) - 配置调整的step间隔。

    异常：
        - **TypeError** - 当 `interval` 类型不为int。
        - **ValueError** - 当 `interval` 的值小于零。
