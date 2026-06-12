mindspore.dataset.config.get_error_samples_mode
===============================================

.. py:function:: mindspore.dataset.config.get_error_samples_mode()

    获取当前数据管道中处理错误样本的策略配置。如果之前未调用 `set_error_samples_mode`，默认设置为 ``ErrorSamplesMode.RETURN``。

    返回：
        :class:`mindspore.dataset.config.ErrorSamplesMode` ，当前数据管道处理错误样本的方法。

        - ErrorSamplesMode.RETURN: 表示错误样本会导致产生错误并返回。
        - ErrorSamplesMode.REPLACE: 表示错误样本会被内部确定的样本替换。
        - ErrorSamplesMode.SKIP: 表示错误样本会被跳过。
