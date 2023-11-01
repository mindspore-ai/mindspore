mindspore.experimental.optim.RAdam
===================================

.. py:class:: mindspore.experimental.optim.RAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.)

    RAdam 算法的实现。

    更新公式如下：

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \beta_1, \beta_2
                \text{ (betas)}, \: \theta_0 \text{ (params)}, \:f(\theta) \text{ (objective)}, \:
                \lambda \text{ (weightdecay)},                                                   \\
            &\hspace{13mm} \epsilon \text{ (epsilon)}                                            \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0 \leftarrow 0 \text{ ( second moment)},                                       \\
            &\hspace{18mm} \rho_{\infty} \leftarrow 2/(1-\beta_2) -1                      \\[-1.ex]
            &\rule{110mm}{0.4pt}  \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{6mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{6mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{6mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{6mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{6mm}\rho_t \leftarrow \rho_{\infty} -
                2 t \beta^t_2 /\big(1-\beta_2^t \big)                                    \\[0.1.ex]
            &\hspace{6mm}\textbf{if} \: \rho_t > 5                                               \\
            &\hspace{12mm} l_t \leftarrow \frac{\sqrt{ (1-\beta^t_2) }}{ \sqrt{v_t} +\epsilon  } \\
            &\hspace{12mm} r_t \leftarrow
      \sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_{\infty}}{(\rho_{\infty}-4)(\rho_{\infty}-2) \rho_t}} \\
            &\hspace{12mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t} r_t l_t        \\
            &\hspace{6mm}\textbf{else}                                                           \\
            &\hspace{12mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``1e-3``。
        - **betas** (Tuple[float, float], 可选) - 梯度及其平方的运行平均值的系数。默认值：``(0.9, 0.999)``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。必须大于0。默认值：``1e-8``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``0.0``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `eps` 小于0。
        - **ValueError** - `weight_decay` 小于0。
        - **ValueError** - `betas` 内元素的取值范围不在0-1之间。
