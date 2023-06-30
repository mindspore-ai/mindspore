mindspore.nn.StepLR
==================================

.. py:class:: mindspore.nn.StepLR(optimizer, step_size, gamma=0.5, last_epoch=-1, verbose=False)

    每 `step_size` 个epoch按 `gamma` 衰减每个参数组的学习率。`StepLR` 对于学习率的衰减可能与外部对于学习率的改变同时发生。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `nn.optim_ex` 下的优化器配合使用。

    参数：
        - **optimizer** (Optimizer) - 优化器实例。
        - **step_size** (int) - 学习率衰减的周期。
        - **gamma** (float，可选) -  学习率衰减的乘法因子。默认值: ``0.1``。
        - **last_epoch** (int，可选) - epoch/step数。默认值： ``-1``。
        - **verbose** (bool，可选) - 是否打印学习率。默认值： ``False``。