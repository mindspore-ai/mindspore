mindspore.experimental.optim.lr_scheduler.CosineAnnealingWarmRestarts
======================================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1)

    使用余弦退火热重启对优化器参数组的学习率进行改变。下述公式中， :math:`\eta_{max}` 为初始学习率，:math:`\eta_{min}` 为学习率变化的最小值，:math:`\eta_{t}` 为当前学习率，:math:`T_{0}` 为初始周期，:math:`T_{i}` 为当前周期，即SGDR两次热重启之间的迭代数，:math:`T_{cur}` 为当前周期内的迭代数。

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    当 :math:`T_{cur}=T_{i}` 时，:math:`\eta_t = \eta_{min}`，热重启后 :math:`T_{cur}=0` 时，设置 :math:`\eta_t=\eta_{max}`。

    详情请查看 `SGDR: Stochastic Gradient Descent with Warm Restarts <https://arxiv.org/abs/1608.03983>`_。

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **T_0** (int) - 余弦函数的初始周期数。
        - **T_mult** (int, 可选) - 迭代中对 :math:`T_{i}` 进行增长的乘法系数。默认值：``1``。
        - **eta_min** (Union(float, int), 可选) - 学习率的最小值。默认值： ``0``。
        - **last_epoch** (int，可选) - 当前scheduler的 `step()` 方法的执行次数。默认值： ``-1``。

    异常：
        - **ValueError** - `T_0` 小于等于0或不是int类型。
        - **ValueError** - `T_mult` 小于等于1或不是int类型。
        - **ValueError** - `eta_min` 不是int或float类型。

    .. py:method:: step(epoch=None)

        按照定义的计算逻辑计算并修改学习率。

        参数：
            - **epoch** (int，可选) - epoch数。默认值： ``None``。
