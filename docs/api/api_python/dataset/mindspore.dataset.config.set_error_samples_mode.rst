mindspore.dataset.config.set_error_samples_mode
===============================================

.. py:function:: mindspore.dataset.config.set_error_samples_mode(error_samples_mode)

    设置在数据管道中处理错误样本的策略。

    .. note::
        - 此错误样本特性仅适用于数据集管道中的Map操作。
        - 对于 'ErrorSamplesMode.REPLACE' 模式，将使用内部缓存中的其他样本。
        - 如果在多机设置中使用 'ErrorSamplesMode.SKIP' 模式，请手动确保每个分片的有效样本数是相同的（否则可能会导致挂起）。一种解决方法是通过Concat操作拼接一个样本全有效的数据集，然后采用Take操作填补跳过的错误样本数。

    参数：
        - **error_samples_mode** (:class:`mindspore.dataset.config.ErrorSamplesMode`) - 处理错误样本的方法。默认值：ErrorSamplesMode.RETURN。

          - ErrorSamplesMode.RETURN：表示处理过程中遇到错误样本时将报错并抛出异常。
          - ErrorSamplesMode.REPLACE：表示处理过程中遇到错误样本时将使用正确的样本替换处理。
          - ErrorSamplesMode.SKIP：表示处理过程中遇到错误样本时将直接跳过此样本。

    异常：
        - **TypeError** - `error_samples_mode` 不是 :class:`mindspore.dataset.config.ErrorSamplesMode` 类型。
