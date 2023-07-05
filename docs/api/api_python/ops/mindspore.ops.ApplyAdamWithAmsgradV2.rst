mindspore.ops.ApplyAdamWithAmsgradV2
=====================================

.. py:class:: mindspore.ops.ApplyAdamWithAmsgradV2(use_locking=False)

    根据Adam算法更新变量var。

    更新公式如下：

    .. math::
        \begin{array}{l1} \\
            lr_t:=learning\_rate*\sqrt{1-\beta_2^t}/(1-\beta_1^t) \\
            m_t:=\beta_1*m_{t-1}+(1-\beta_1)*g \\
            v_t:=\beta_2*v_{t-1}+(1-\beta_2)*g*g \\
            \hat v_t:=\max(\hat v_{t-1}, v_t) \\
            var:=var-lr_t*m_t/(\sqrt{\hat v_t}+\epsilon) \\
        \end{array}

    所有输入符合隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    参数：
        - **use_locking** (bool) - 如果为 ``True`` ， `var` ， `m` 和 `v` 的更新将受到锁的保护。否则，行为为未定义，很可能出现较少的冲突。默认值为 ``False`` 。

    输入：
        - **var** (Parameter) - 待更新的网络参数，为任意维度。数据类型为float16、float32或float64。
        - **m** (Parameter) - 一阶矩，shape与 `var` 相同。
        - **v** (Parameter) - 二阶矩。shape与 `var` 相同。
        - **vhat** (Parameter) - 公式中的 :math:`\hat v_t` 。shape和类型与 `var` 相同。
        - **beta1_power** (Union[float, Tensor]) - 公式中的 :math:`beta_1^t(\beta_1^{t})` ，数据类型为float16、float32或float64。
        - **beta2_power** (Union[float, Tensor]) - 公式中的 :math:`beta_2^t(\beta_2^{t})` ，数据类型为float16、float32或float64。
        - **lr** (Union[float, Tensor]) - 学习率。数据类型为float16、float32或float64的Tensor。
        - **beta1** (Union[float, Tensor]) - 一阶矩的指数衰减率。数据类型为float16、float32或float64。
        - **beta2** (Union[float, Tensor]) - 二阶矩的指数衰减率。数据类型为float16、float32或float64。
        - **epsilon** (Union[float, Tensor]) - 加在分母上的值，以确保数值稳定。数据类型为float16、float32或float64。
        - **grad** (Tensor) - 为梯度，shape与 `var` 相同。

    输出：
        4个Tensor组成的tuple，更新后的数据。

        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **m** (Tensor) - shape和数据类型与 `m` 相同。
        - **v** (Tensor) - shape和数据类型与 `v` 相同。
        - **vhat** (Tensor) - shape和数据类型与 `vhat` 相同。

    异常：
        - **TypeError** - 如果 `var` 、 `m` 、 `v` 、 `vhat` 不是Parameter。
        - **TypeError** - 如果 `var` 、 `m` 、 `v` 、 `vhat` 、`beta1_power` 、 `beta2_power` 、 `lr` 、 `beta1` 、 `beta2` 、 `epsilon` 或 `grad` 的数据类型既不是float16也不是float32，也不是float64。
        - **RuntimeError** - 如果 `var` 、 `m` 、 `v` 、 `vhat` 和 `grad` 不支持数据类型转换。
