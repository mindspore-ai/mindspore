mindspore.mint.optim.AdamW
===================================

.. py:class:: mindspore.mint.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False, *, maximize=False)

    Adaptive Moment Estimation Weight Decay(AdamW)算法的实现。

    .. math::
        \begin{aligned}
                &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                    \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                    \: \epsilon \text{ (epsilon)}                                                    \\
                &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                    \: \textit{maximize}                                                             \\
                &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                    \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
                &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
                &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
                &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
                &\hspace{5mm}\textbf{else}                                                           \\
                &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
                &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
                &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
                &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
                &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
                &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
                &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
                &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                    \widehat{v_t})                                                                   \\
                &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                    \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
                &\hspace{5mm}\textbf{else}                                                           \\
                &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                    \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
                &\bf{return} \:  \theta_t                                                     \\[-1.ex]
        \end{aligned}

    .. warning::
        - 这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 
        - 对于Ascend，仅Atlas A2以上平台支持。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[float, Tensor], 可选) - 学习率。默认值：``1e-3``。
        - **betas** (Tuple[float, float], 可选) - 动量矩阵的指数衰减率。默认值：``(0.9, 0.999)``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。必须大于0。默认值：``1e-8``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``1e-2``。
        - **amsgrad** (bool, 可选) - 是否使用AMSGrad算法。默认值：``False``。

    关键字参数：
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor], 可选) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `eps` 小于0。
        - **ValueError** - `betas` 范围不在[0, 1)之间。
        - **ValueError** - `weight_decay` 小于0。