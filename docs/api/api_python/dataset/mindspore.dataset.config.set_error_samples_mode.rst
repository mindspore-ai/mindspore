mindspore.dataset.config.set_error_samples_mode
===============================================

.. py:function:: mindspore.dataset.config.set_error_samples_mode(error_samples_mode)

    设置在数据管道中处理错误样本的方法。

    .. note::
        - 此错误样本特性仅适用于数据集管道中的Map操作。
        - 对于替换模式，缓存将会用于存放内部确定的样本。
        - 如果在多机设置中使用跳过模式，请手动确保每个分片的有效样本数是相同的（否则可能会导致挂起）。一种解决方法是手动Concat一个样本全有效的数据集，然后采用Take操作填补跳过的错误样本数。

    参数：
        - **error_samples_mode** (ErrorSamplesMode) - 处理错误样本的方法。默认值：ErrorSamplesMode.RETURN。

          - ErrorSamplesMode.RETURN：表示错误样本会导致产生错误并返回。
          - ErrorSamplesMode.REPLACE：表示错误样本会被内部确定的样本替换。
          - ErrorSamplesMode.SKIP：表示错误样本会被跳过。

    异常：
        - **TypeError** - `error_samples_mode` 不是ErrorSamplesMode类型。
