mindspore.experimental.optim.lr_scheduler.ReduceLROnPlateau
============================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)

    当指标停止改进时降低学习率。训练中学习停滞情况下，模型通常会受益于将学习率降低2-10倍。该调度程序在执行过程中读取 `step` 方法中传入的指标 `metrics`，如果在 `patience` 的时期内没有得到改进，则学习率会降低。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **mode** (str, 可选) - 'min' 或 'max'。在 'min' 模式下，当监控指标不再下降，降低学习率；在 'max' 模式下，当监控指标不再上升，降低学习率。默认值：'min'。
        - **factor** (float, 可选) - 学习率衰减因子。默认值：``0.1``。
        - **patience** (int, 可选) - 评估指标无改善情况下等待的epoch数。例如，如果 `patience = 2`，则前2个无改善的epoch将被忽略，从第3个epoch降低学习率。默认值：``10``。
        - **threshold** (float, 可选) - 衡量指标变好的最小阈值。默认值：``1e-4``。
        - **threshold_mode** (str, 可选) - ``'rel'`` 或 ``'abs'``。每次评估指标是否变好的基准为 dynamic_threshold，在 ``'rel'`` 模式下，如果 `mode` 为 ``'max'``，则 dynamic_threshold = best * ( 1 + threshold )； 如果 `mode` 为 ``'min'``，则 best * ( 1 - threshold )。``'abs'`` 模式下，如果 `mode` 为 ``'max'``，则 dynamic_threshold = best + threshold；如果 `mode` 为 ``'min'``，则 best - threshold。默认值： ``'rel'``。
        - **cooldown** (int, 可选) - 在降低学习率后恢复正常运行之前要等待的epoch数。默认值：``0``。
        - **min_lr** (Union(float, list), 可选) - 标量或标量列表，所有参数组或每个组的学习率最小值。默认值：``0``。
        - **eps** (float, 可选) - 应用于学习率的最小衰减。如果学习率变化的差异小于 `eps`，则忽略更新。默认值：``1e-8``。


    异常：
        - **ValueError** - `factor` 大于等于1。
        - **TypeError** - `optimizer` 不是 `Optimizer`。
        - **ValueError** - `min_lr` 为list或tuple时，其长度不等于参数组数目。
        - **ValueError** - `mode` 不是 ``'min'`` 或 ``'max'``。
        - **ValueError** - `threshold_mode` 不是 ``'rel'`` 或 ``'abs'``。

    .. py:method:: get_last_lr()

        返回当前使用的学习率。

    .. py:method:: in_cooldown()

        是否在 `cooldown` 时期。

    .. py:method:: step(metrics)

        按照定义的计算逻辑计算并修改学习率。

        参数：
            - **metrics** (Union(int, float)) - 评估指标值。