mindspore.experimental.optim.lr_scheduler.CosineAnnealingLR
=============================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)

    使用余弦退火对优化器参数组的学习率进行改变。下述公式中， :math:`\eta_{max}` 为初始学习率，:math:`\eta_{min}` 为学习率变化的最小值，:math:`\T_{max}` 为余弦函数的半周期，:math:`\T_{cur}` 为当前周期内的迭代数，:math:`\eta_{t}` 为当前学习率。

    .. math::
        \begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}

    详情请查看 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **T_max** (int) - 余弦函数的半周期。
        - **eta_min** (float, 可选) - 学习率的最小值。默认值：``0``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值：``-1``。