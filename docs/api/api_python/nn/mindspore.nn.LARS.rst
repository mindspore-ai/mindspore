mindspore.nn.LARS
==================

.. py:class:: mindspore.nn.LARS(*args, **kwargs)

    LARS算法的实现。

    LARS算法采用大量的优化技术。详见论文 `LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_。

    更新公式如下：

    .. math::

        \begin{array}{ll} \\
            \lambda  = \frac{\theta  \text{ * } || \omega  ||  } \\
                            {|| g_{t} || \text{ + } \delta \text{ * } || \omega  || }  \\
            \lambda  =
            \begin{cases}
                \min(\frac{\lambda}{\alpha }, 1)
                    & \text{ if } clip = True \\
                \lambda
                    & \text{ otherwise }
            \end{cases}\\
            g_{t+1} = \lambda * (g_{t} + \delta * \omega)
        \end{array}

    :math:`\theta` 表示 `coefficient` ，:math:`\omega` 表示网络参数，:math:`g` 表示 `gradients`，:math:`t` 表示当前step，:math:`\delta` 表示 `optimizer` 配置的 `weight_decay` ，:math:`\alpha` 表示 `optimizer` 配置的 `learning_rate` ，:math:`clip` 表示 `use_clip`。


    **参数：**

    - **optimizer** (Optimizer) - 待封装和修改梯度的MindSpore优化器。
    - **epsilon** (float) - 将添加到分母中，提高数值稳定性。默认值：1e-05。
    - **coefficient** (float) - 计算局部学习速率的信任系数。默认值：0.001。
    - **use_clip** (bool) - 计算局部学习速率时是否裁剪。默认值：False。
    - **lars_filter** (Function) - 用于指定使用LARS算法的网络参数。默认值：lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name。

    **输入：**

    - **gradients** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。


    **输出：**

    Union[Tensor[bool], tuple[Parameter]]，取决于 `optimizer` 的输出。

    **支持平台：**

    ``Ascend`` ``CPU``

    **样例：**

    >>> net = Net()
    >>> loss = nn.SoftmaxCrossEntropyWithLogits()
    >>> opt = nn.Momentum(net.trainable_params(), 0.1, 0.9)
    >>> opt_lars = nn.LARS(opt, epsilon=1e-08, coefficient=0.02)
    >>> model = Model(net, loss_fn=loss, optimizer=opt_lars, metrics=None)
