mindspore.nn.LARS
==================

.. py:class:: mindspore.nn.LARS(optimizer, epsilon=1e-05, coefficient=0.001, use_clip=False, lars_filter=lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name)

    LARS算法的实现。

    LARS算法采用大量的优化技术。详见论文 `LARGE BATCH TRAINING OF CONVOLUTIONAL NETWORKS <https://arxiv.org/abs/1708.03888>`_。

    更新公式如下：

    .. math::
        \begin{array}{ll} \\
            &\newline
            &\hline \\
            &\textbf{Parameters}: \text{base learning rate } \gamma_{0} , \text{ momentum  m}, \text{ weight decay }
             \lambda , \\
            &\hspace{5mm}\text{ LARS coefficient } \eta , \text{ number of steps } T \\
            &\textbf{Init}: \text{ t=0, v=0, init weight }  w_{0}^{l}  \text{ for each layer } l \\[-1.ex]
            &\newline
            &\hline \\
            &\textbf{while} \text{ t<T  for each layer } l \textbf{ do} \\
            &\hspace{5mm}g_{t}^{l} \leftarrow \nabla L\left(w_{t}^{l}\right) \\
            &\hspace{5mm}\gamma_{t} \leftarrow \gamma_{0} *\left(1-\frac{t}{T}\right)^{2} \\
            &\hspace{5mm}\gamma^{l} \leftarrow \eta *\frac{\left\|w_{t}^{l}\right\|}{\left\|g_{t}^{l}\right\|+
             \lambda\left\|w_{t}^{l}\right\|} \text{(compute the local LR } \gamma^{ l)} \\
            &\hspace{5mm}v_{t+1}^{l} \leftarrow m v_{t}^{l}+\gamma_{t+1} * \gamma^{l} *\left(g_{t}^{l}+\lambda
             w_{t}^{l}\right) \\
            &\hspace{5mm}w_{t+1}^{l} \leftarrow w_{t}^{l}-v_{t+1}^{l} \\
            &\textbf{ end while } \\[-1.ex]
            &\newline
            &\hline \\[-1.ex]
        \end{array}

    :math:`w` 表示 `params`，:math:`g` 表示 `gradients` ，:math:`t` 表示当前step，:math:`\lambda` 表示 `optimizer` 配置的 `weight_decay` ，:math:`\gamma` 表示 `optimizer` 配置的 `learning_rate` ，:math:`\eta` 表示 `coefficient` 。

    参数：
        - **optimizer** (:class:`mindspore.nn.Optimizer`) - 待封装和修改梯度的MindSpore优化器。
        - **epsilon** (float) - 将添加到分母中，提高数值稳定性。默认值： ``1e-05`` 。
        - **coefficient** (float) - 计算局部学习速率的信任系数。默认值： ``0.001`` 。
        - **use_clip** (bool) - 计算局部学习速率时是否裁剪。默认值： ``False`` 。
        - **lars_filter** (Function) - 用于指定使用LARS算法的网络参数。默认值：lambda x: 'LayerNorm' not in x.name and 'bias' not in x.name。

    输入：
        - **gradients** (tuple[Tensor]) - 优化器中 `params` 的梯度，shape与优化器中的 `params` 相同。

    输出：
        Union[Tensor[bool]，tuple[Parameter]]，取决于 `optimizer` 的输出。
