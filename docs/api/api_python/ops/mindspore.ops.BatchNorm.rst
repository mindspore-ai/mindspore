mindspore.ops.BatchNorm
========================

.. py:class:: mindspore.ops.BatchNorm(is_training=False, epsilon=1e-5, momentum=0.1, data_format="NCHW")

    对输入数据进行归一化(Batch Normalization)和更新参数。

    批量归一化广泛应用于卷积神经网络中。此运算对输入应用归一化，避免内部协变量偏移，详见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和学习参数进行训练，学习的参数见如下公式中，

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    其中， :math:`\gamma` 为 `scale`， :math:`\beta` 为 `bias`， :math:`\epsilon` 为 `epsilon`， :math:`mean` 为 :math:`x` 的均值， :math:`variance` 为 :math:`x` 的方差。

    .. warning::
        - 如果该运算用于推理，并且输出"reserve_space_1"和"reserve_space_2"可用，则"reserve_space_1"的值与"mean"相同，"reserve_space_2"的值与"variance"相同。
        - 对于Ascend 310，由于平方根指令，结果精度未能达到1‰。

    参数：
        - **is_training** (bool) - 如果 `is_training` 为True，则在训练期间计算 `mean` 和 `variance`。如果 `is_training` 为False，则在推理期间从checkpoint加载。默认值：False。
        - **epsilon** (float) - 添加到分母上的值，以确保数值稳定性。默认值：1e-5。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。（例如 :math:`new\_running\_mean = (1 - momentum) * running\_mean + momentum * current\_mean`）。动量值必须为[0, 1]。默认值：0.1。
        - **data_format** (str) - 输入数据格式，可选值有：'NHWC'或'NCHW'，'NHWC'仅在GPU上支持。默认值：'NCHW'。

    输入：
        如果 `is_training` 为False，则输入为多个Tensor。

        - **input_x** (Tensor) - 数据输入，shape为 :math:`(N, C)` 的Tensor，数据类型为float16或float32。
        - **scale** (Tensor) - 输入Scalar，shape为 :math:`(C,)` 的Tensor，数据类型为float16或float32。
        - **bias** (Tensor) - 输入偏置项，shape为 :math:`(C,)` 的Tensor，具有与 `scale` 相同的数据类型。
        - **mean** (Tensor) - 输入均值，shape为 :math:`(C,)` 的Tensor，具有与 `scale` 相同的数据类型。
        - **variance** (Tensor) - 输入方差，shape为 :math:`(C,)` 的Tensor，具有与 `scale` 相同的数据类型。

        如果 `is_training` 为True，则 `scale` 、 `bias` 、 `mean` 和 `variance` 是Parameter。

        - **input_x** (Tensor) - 数据输入，shape为 :math:`(N, C)` 的Tensor，数据类型为float16或float32。
        - **scale** (Parameter) - 输入Scalar，shape为 :math:`(C,)` 的参数，数据类型为float16或float32。
        - **bias** (Parameter) - 输入偏置项，shape为 :math:`(C,)` 的参数，具有与 `scale` 相同的数据类型。
        - **mean** (Parameter) - 输入均值，shape为 :math:`(C,)` 的参数，具有与 `scale` 相同的数据类型。
        - **variance** (Parameter) - 输入方差，shape为 :math:`(C,)` 的参数，具有与 `scale` 相同的数据类型。
 
    输出：
        5个Tensor组成的tuple、归一化输入和更新的参数。

        - **output_x** (Tensor) - 数据类型和shape与输入 `input_x` 相同。shape为 :math:`(N, C)` 。
        - **batch_mean** (Tensor) - 输入的均值，shape为 :math:`(C,)` 的一维Tensor。
        - **batch_variance** (Tensor) - 输入的方差，shape为 :math:`(C,)` 的一维Tensor。
        - **reserve_space_1** (Tensor) - 需要计算梯度时，被重新使用的均值，shape为 :math:`(C,)` 的一维Tensor。
        - **reserve_space_2** (Tensor) - 需要计算梯度时，被重新使用的方差，shape为 :math:`(C,)` 的一维Tensor。

    异常：
        - **TypeError：** `is_training` 不是bool。
        - **TypeError：** `epsilon` 或 `momentum` 的数据类型不是float。
        - **TypeError：** `data_format` 不是str。
        - **TypeError：** `input_x`、`scale`、`bias`、`mean` 或  `variance` 不是Tensor。
        - **TypeError：** `input_x` 和 `scale` 的数据类型既不是float16，也不是float32。 