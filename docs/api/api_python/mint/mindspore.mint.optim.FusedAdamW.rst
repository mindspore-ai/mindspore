mindspore.mint.optim.FusedAdamW
===================================

.. py:class:: mindspore.mint.optim.FusedAdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, maximize=False)

    Fused Adaptive Moment Estimation Weight Decay（FusedAdamW）算法的实现。

    .. math::
        \begin{array}{l}
                &\newline
                &\hline \\
                &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                    \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                    \: \epsilon \text{ (epsilon)}                                                    \\
                &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                    \: \textit{maximize}                                                             \\
                &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                    \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
                &\newline
                &\hline \\
                &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
                &\hspace{6mm}\textbf{if} \: \textit{maximize}:                                       \\
                &\hspace{11mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
                &\hspace{6mm}\textbf{else}                                                           \\
                &\hspace{11mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
                &\hspace{6mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
                &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
                &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
                &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
                &\hspace{6mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
                &\hspace{6mm}\textbf{if} \: amsgrad                                                  \\
                &\hspace{11mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                    \widehat{v_t})                                                                   \\
                &\hspace{11mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                    \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
                &\hspace{6mm}\textbf{else}                                                           \\
                &\hspace{11mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                    \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
                &\newline
                &\hline \\[-1.ex]
                &\bf{return} \:  \theta_t                                                     \\[-1.ex]
                &\newline
                &\hline \\[-1.ex]
        \end{array}

    更多关于AdamW算法的细节可以参考论文 `Decoupled Weight Decay Regularization <https://arxiv.org/abs/1711.05101>`_ 和
    `On the Convergence of Adam and Beyond <https://openreview.net/forum?id=ryQu7f-RZ>`_。

    .. warning::
        - 这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。
        - `amsgrad` 参数必须在所有参数组中保持一致。不支持在不同的参数组中混用 `amsgrad=True` 和 `amsgrad=False`。
        - 对于Ascend，只支持Atlas A2系列产品。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (float, 可选) - 学习率。默认值： ``1e-3``。
        - **betas** (Tuple[float, float], 可选) - 动量矩阵的指数衰减率。默认值： ``(0.9, 0.999)``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。必须大于0。默认值： ``1e-8``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值： ``1e-2``。
        - **amsgrad** (bool, 可选) - 是否使用AMSGrad算法。默认值： ``False``。该参数必须在所有参数组中保持一致。

    关键字参数：
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值： ``False``。

    输入：
        - **gradients** (tuple[Tensor]) - `params` 的梯度。

    异常：
        - **ValueError** - `betas` 范围不在[0, 1)之间。
        - **ValueError** - `weight_decay` 小于0。
        - **ValueError** - 不同参数组的 `amsgrad` 不一致。