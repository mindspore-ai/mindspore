mindspore.experimental.optim.lr_scheduler.OneCycleLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, total_steps=None, epochs=None, steps_per_epoch=None, pct_start=0.3, anneal_strategy='cos', div_factor=25., final_div_factor=1e4, three_phase=False, last_epoch=-1, verbose=False)

    按照1cycle学习率策略设置各参数组的学习率。 1cycle 策略将学习率从初始学习率增加到某个最大学习率，然后从该最大学习率退火到远低于初始学习率的某个最小学习率。详情请参考论文 `Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates <https://arxiv.org/abs/1708.07120>`_。

    一个循环周期中的步骤数将通过以下两种方式之一确定（按优先顺序排列）：

    - 设定 `total_steps` 的值。

    - 设定了epoch数 `epochs` 和每个epoch的步骤数 `steps_per_epoch`。在这种情况下，总步数通过 `total_steps = epochs * steps_per_epoch` 进行计算。

    请务必设定 `total_steps` 的值，或设定 `epochs` 和 `steps_per_epoch` 的值。

    该调度程序的默认行为遵循 1cycle 的 fastai 实现，该实现声称'未发布的工作仅使用两个阶段就显示出了更好的结果'。如需模仿原始论文的行为，请设置 `Three_phase=True`。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **max_lr** (Union(float, list)) - 每个参数组的学习率上界值。
        - **total_steps** (int，可选) - 循环中的总步数。如果此参数未设定值，则需要通过通过 `epochs` 和 `steps_per_epoch` 的值来计算。默认值： ``None``。
        - **epochs** (int，可选) - 训练的epoch数。如果未设定 `total_steps` 的值，则将其与 `steps_per_epoch` 一起使用，以用来计算循环中的总步数。默认值： ``None``。
        - **steps_per_epoch** (int，可选) - 每个epoch中训练的步数。如果未设定 `total_steps` 的值，则将其与 `epoch` 一起使用，以便计算循环中的总步数。默认值： ``None`` 。
        - **pct_start** (float，可选) - 一个循环周期内用于学习率增长（的步数）占用的百分比。默认值： ``0.3``。
        - **anneal_strategy** (str，可选) - 退火策略，可设置 ``'cos'`` 或 ``'linear'``。 ``'cos'`` 表示余弦退火， ``'Linear'`` 表示线性退火。默认值： ``'cos'``。
        - **div_factor** (float，可选) - 除法因子。按照 `initial_lr = max_lr/div_factor` 确定初始学习率。默认值： ``25``。
        - **final_div_factor** (float，可选) - 最终的除法因子。按照 `min_lr = initial_lr/final_div_factor` 确定最小学习率。默认值： ``1e4``。
        - **three_phase** (bool，可选) - 如果为 ``True``，则使 three_phase 策略调整学习率，否则使用 two_phase 策略，具体算法细节请参考上述论文。默认值： ``False``。
        - **last_epoch** (int，可选) - epoch/step数。默认值： ``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值： ``False``。
