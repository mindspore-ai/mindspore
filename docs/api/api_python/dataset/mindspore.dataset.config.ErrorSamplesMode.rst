mindspore.dataset.config.ErrorSamplesMode
===========================================

.. py:class:: mindspore.dataset.config.ErrorSamplesMode

    指定数据管道中处理错误样本的策略。

    - **ErrorSamplesMode.RETURN** - 表示处理过程中遇到错误样本时将报错并抛出异常。
    - **ErrorSamplesMode.REPLACE** - 表示处理过程中遇到错误样本时将使用正确的样本替换处理。
    - **ErrorSamplesMode.SKIP** - 表示处理过程中遇到错误样本时将直接跳过此样本。
