mindspore.nn.AdaSumByGradWrapCell
=================================

.. py:class:: mindspore.nn.AdaSumByGradWrapCell(optimizer)

    Adaptive Summation (AdaSum)算法的实现，根据梯度计算。应用于semi_auto_parallel/auto_parallel模式。

    请参阅论文 `AdaSum: Scaling Distributed Training with Adaptive Summation <https://arxiv.org/abs/2006.02924>`_。

    公式如下：

    .. math::
        \begin{array}{ll}
          w_{t+1}=w_{t} - \alpha \cdot Adasum(g_{1}, g_{2})  \\
          w_{t+1}=w_{t} - \alpha \cdot [(1 - \frac{g_2^{T}\cdot g_1}{2\cdot \left \| g_1 \right \|^2 })\cdot g_1 +  (1 - \frac{g_1^{T}\cdot g_2}{2\cdot \left \| g_2 \right \|^2 })\cdot g_2]  \\
        \end{array}

    在本实现中， :math:`g` 代表权重的梯度，下标代表数据并行维度下不同的设备。

    .. note::
        本接口推荐应用于半自动并行或者全自动并行模式。针对数据并行模式，推荐使用mindspore.boost功能以使用AdaSum。
        使用本接口时，训练的卡的数量必须是2的幂，并且至少需要16张卡。目前，使用本接口时不支持优化器并行和流水线并行。

    参数：
        - **optimizer** (Union[Cell]) - 用于更新权重的优化器。优化器的构造函数只允许一个输入。

    输入：
        - **grads** (tuple[Tensor]) - `params` 的梯度，形状（shape）与 `params` 相同，与所传优化器的输入一致。

    异常：
        - **RuntimeError** - `parallel_mode` 使用了 `stand_alone` 模式， AdaSum仅支持在分布式场景下使用。
        - **RuntimeError** - 同时使用了优化器并行，暂时不支持在优化器并行场景下使用AdaSum。
        - **RuntimeError** - 同时使用了流水线并行，暂时不支持在流水线并行场景下使用AdaSum。
        - **RuntimeError** - `device_num` 不是2的幂，或者小于16。
