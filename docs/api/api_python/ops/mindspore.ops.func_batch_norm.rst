mindspore.ops.batch_norm
========================

.. py:function:: mindspore.ops.batch_norm(input_x, running_mean, running_var, weight, bias, training=False, momentum=0.1, eps=1e-5)

    对输入数据进行批量归一化和更新参数。

    批量归一化广泛应用于卷积神经网络中。此运算对输入应用归一化，避免内部协变量偏移，详见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和学习参数进行训练，学习的参数见如下公式中，

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    其中， :math:`\gamma` 为 `weight`， :math:`\beta` 为 `bias`， :math:`\epsilon` 为 `eps`， :math:`mean` 为 :math:`x` 的均值， :math:`variance` 为 :math:`x` 的方差。

    .. warning::
        - 对于Ascend 310，由于平方根指令，结果精度未能达到1‰。

    .. note::
        - 如果 `training` 为False，则 `running_mean` 、 `running_var` 、 `weight` 和 `bias` 是Tensor。
        - 如果 `training` 为True，则 `running_mean` 、 `running_var` 、 `weight` 和 `bias` 是Parameter。

    参数：
        - **input_x** (Tensor) - 数据输入，shape为 :math:`(N, C)` 的Tensor，数据类型为float16或float32。
        - **running_mean** (Union[Tensor, Parameter]) - shape为 :math:`(C,)` ，具有与 `weight` 相同的数据类型。
        - **running_var** (Union[Tensor, Parameter]) - shape为 :math:`(C,)` ，具有与 `weight` 相同的数据类型。
        - **weight** (Union[Tensor, Parameter]) - shape为 :math:`(C,)` ，数据类型为float16或float32。
        - **bias** (Union[Tensor, Parameter]) - shape为 :math:`(C,)` ，具有与 `weight` 相同的数据类型。
        - **training** (bool, 可选) - 如果 `training` 为 `True`，`running_mean` 和 `running_var` 会在训练过程中进行计算。
          如果 `training` 为 `False` ，它们会在推理阶段从checkpoint中加载。默认值：False。
        - **momentum** (float, 可选) - 动态均值和动态方差所使用的动量。（例如 :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`）。动量值必须为[0, 1]。默认值：0.1。
        - **eps** (float, 可选) - 添加到分母上的值，以确保数值稳定性。默认值：1e-5。

    返回：
        Tensor，数据类型与shape大小与 `input_x` 相同，其中，shape大小为 :math:`(N, C)` 。

    异常：
        - **TypeError** - `training` 不是bool。
        - **TypeError** - `eps` 或 `momentum` 的数据类型不是float。
        - **TypeError** - `input_x`、`weight`、`bias`、`running_mean` 或  `running_var` 不是Tensor。
        - **TypeError** - `input_x` 和 `weight` 的数据类型既不是float16，也不是float32。
