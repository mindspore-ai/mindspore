mindspore.experimental.optim.Adadelta
=======================================

.. py:class:: mindspore.experimental.optim.Adadelta(params, lr=1., rho=0.9, eps=1e-6, weight_decay=0., *, maximize=False)

    Adadelta算法的实现。

    更新公式如下：

    .. math::
       \begin{aligned}
            &\rule{150mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    .. warning::
        这是一个实验性的优化器接口，需要和 `LRScheduler <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.experimental.html#lrscheduler%E7%B1%BB>`_ 下的动态学习率接口配合使用。

    参数：
        - **params** (Union[list(Parameter), list(dict)]) - 网络参数的列表或指定了参数组的列表。
        - **lr** (Union[int, float, Tensor], 可选) - 学习率。默认值：``1.0``。
        - **rho** (float, 可选) - 梯度平方平均值的系数。上述公式中的 :math:`\rho`。默认值：``0.9``。
        - **eps** (float, 可选) - 加在分母上的值，以确保数值稳定。上述公式中的 :math:`\epsilon`。默认值：``1e-6``。
        - **weight_decay** (float, 可选) - 权重衰减（L2 penalty）。默认值：``0.``。

    关键字参数：
        - **maximize** (bool, 可选) - 是否根据目标函数最大化网络参数。默认值：``False``。

    输入：
        - **gradients** (tuple[Tensor]) - 网络权重的梯度。

    异常：
        - **ValueError** - 学习率不是int、float或Tensor。
        - **ValueError** - 学习率小于0。
        - **ValueError** - `eps` 小于0。
        - **ValueError** - `rho` 范围不在0-1之间。
        - **ValueError** - `weight_decay` 小于0。