优化器和混合精度之间通常没有联系。但是，当使用 `FixedLossScaleManager` 且 `FixedLossScaleManager` 中的 `drop_overflow_update` 设置为False时，优化器需要设置'loss_scale'。

由于此优化器没有 `loss_scale` 的参数，因此需要通过其他方式处理 `loss_scale` 。

如何正确处理 `loss_scale` 详见 `LossScale <https://www.mindspore.cn/tutorials/experts/zh-CN/r1.9/others/mixed_precision.html>`_。
