mindspore.experimental.optim.lr_scheduler.PolynomialLR
=======================================================

.. py:class:: mindspore.experimental.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5, power=1.0, last_epoch=-1)

    每个epoch，学习率通过多项式拟合来调整。当epoch大于等于 `total_iters` 时，学习率设置为 ``0`` 。注意，这种衰减可能与外部对于学习率的改变同时发生。

    学习率计算的多项式公式如下：

    .. math::
        \begin{split}
        &factor = (\frac{1.0 - \frac{last\_epoch}{total\_iters}}{1.0 - \frac{last\_epoch - 1.0}{total\_iters}})
        ^{power}\\
        &lr = lr \times factor
        \end{split}

    .. warning::
        这是一个实验性的动态学习率接口，需要和 `mindspore.experimental.optim <https://mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#%E5%AE%9E%E9%AA%8C%E6%80%A7%E4%BC%98%E5%8C%96%E5%99%A8>`_ 下的接口配合使用。

    参数：
        - **optimizer** (:class:`mindspore.experimental.optim.Optimizer`) - 优化器实例。
        - **total_iters** (int，可选) - 通过多项式拟合调整学习率的迭代次数。默认值： ``5``。
        - **power** (float，可选) -  多项式的幂。默认值： ``1.0``。
        - **last_epoch** (int，可选) - 最后一个epoch的索引。默认值： ``-1``。
