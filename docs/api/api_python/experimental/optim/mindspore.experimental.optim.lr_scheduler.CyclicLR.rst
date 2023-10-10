mindspore.experimental.optim.lr_scheduler.CyclicLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.CyclicLR(optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle', last_epoch=-1)

    根据循环学习率策略（CLR）设置每个参数组的学习率。该策略以恒定频率在两个边界之间循环学习率值，详情请参考论文 `Cyclical Learning Rates for Training Neural Networks <https://arxiv.org/abs/1506.01186>`_。两个边界之间的距离可以在每次迭代或每个周期的基础上缩放。

    正如论文中提出的，该类（对学习率变化幅度）有三个内置计算策略：

    - "triangular"：没有幅度缩放的基本三角循环。
    - "triangular2"：每个循环将初始幅度缩放一半的基本三角循环。
    - "exp_range": 在每个迭代中按照 :math:`\text{gamma}^{\text{cycle iterations}}` 缩放初始幅度。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **base_lr** (Union(float, list)) - 初始学习率，也是优化器参数组中学习率的下界值。
        - **max_lr** (Union(float, list)) - 每个参数组的学习率上界值。在功能上，（max_lr - base_lr）定义了学习率周期变化的幅度。周期内，当前的学习率的计算方式为base_lr和振幅乘以缩放系数的加和。
        - **step_size_up** (int, 可选) - 递增半周期内的训练迭代次数。默认值：``2000``。
        - **step_size_down** (int, 可选) - 递减半周期内的训练迭代次数。如果 `step_size_down` 为None，则设置为 `step_size_up` 的值。默认值：``None``。
        - **mode** (str, 可选) - "triangular", "triangular2" 或 "exp_range"。对应的计算策略详见上文，如果 `scale_fn` 不是None，则此参数无效。默认值：``"triangular"``。
        - **gamma** (float, 可选) - 'exp_range' 模式下的常量，计算方式为 `gamma**(cycle iterations)`。默认值：``1.0``。
        - **scale_fn** (function, 可选) - 由单个参数的 lambda 匿名函数定义的自定义扩展策略，其中对所有的 `x >= 0`，`0 <= scale_fn（x） <= 1` 。如果设定了此参数，则 `mode` 设定值将被忽略。默认值：``None``。
        - **scale_mode** (str, 可选) - ``'cycle'`` 或 ``'iterations'``。定义 `scale_fn` 是按周期数还是周期内的迭代次数（当前周期内训练迭代的次数）。若传入不合法输入，将默认使用 ``'iterations'`` 模式。默认值： ``'cycle'``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值： ``-1``。

    异常：
        - **ValueError** - `base_lr` 为list或tuple时，长度不等于参数组数目。
        - **ValueError** - `max_lr` 为list或tuple时，长度不等于参数组数目。
        - **ValueError** - `mode` 不是[``'triangular'``, ``'triangular2'``, ``'exp_range'``]且 `scale_fn` 为 ``None``。
