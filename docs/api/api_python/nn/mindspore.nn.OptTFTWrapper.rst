mindspore.nn.OptTFTWrapper
==========================

.. py:class:: mindspore.nn.OptTFTWrapper(opt, **kwargs)

    实现TFT优化器封装器。该封装器将在优化器更新前向MindIO TFT上报状态。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        该优化器依赖于MindIO TFT特性。当前只支持Ascend后端的图模式，并且sink_size的配置必须小于等于1。

    参数：
        - **opt** (Optimizer) - 该参数必须为Optimizer的子类。

    输入：
        - **gradients** (tuple[Tensor]) - 参数opt的 `params` 的梯度，shape与opt的 `params` shape 相同。

    输出：
        Tensor，优化器opt执行返回的结果。

    异常：
        - **TypeError** - 如果opt不是Optimizer的子类。
        - **ValueError** - 如果不是运行在Ascend后端的图模式，或者用户不开启TFT特性。
