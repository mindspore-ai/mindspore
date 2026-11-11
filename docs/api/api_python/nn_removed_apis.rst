旧 nn 层与损失函数移除说明
===========================

本次移除以下 10 个公开类及其重导出，不提供兼容别名。
``nn.layer.dense`` 和 ``nn.layer.timedistributed`` 专属子模块同时移除；
``nn.Dense``、``nn.LossBase``、其他损失函数及共享算子实现保留。
旧网络源码、直接导入和配置中的类名均需同步调整。

.. list-table:: 迁移建议与边界
   :header-rows: 1
   :widths: 25 75

   * - 移除接口（前缀 mindspore.nn）
     - 迁移建议
   * - ``BiDense``
     - 可用自定义 ``Cell`` 实现双线性计算：对每个输出通道分别计算
       ``x1 @ W[k] @ x2.T``（逐样本计算），原层有偏置时再加上 ``bias[k]``。
       需保持权重形状、初始化、参数名称及 checkpoint 映射，不能直接换成单输入 Dense。
   * - ``TimeDistributed``
     - 评估按时间切片调用共享 Cell，或 reshape 后调用再恢复维度。
       必须核对时间轴、输出结构、参数共享、随机数及 BatchNorm 等有状态层的行为；
       reshape 或 vmap 不保证对任意层等价。``nn.wrap.TimeDistributed`` 也已移除。
   * - ``MAELoss``
     - 可评估 ``nn.L1Loss(reduction=...)``，保持原 reduction。
       本仓保留的训练测试使用 L1Loss；调用 construct 的关键字 ``label`` 需改为 ``labels``。
       迁移仍需核对 dtype、广播、归约精度和梯度。
   * - ``RMSELoss``
     - 对原浮点输入可使用自定义 Cell 组合 ``ops.sqrt(nn.MSELoss()(logits, labels))``。
       保持全元素 mean 及广播语义；不要擅自加入 epsilon，需评估零误差处的梯度和输入类型校验。
   * - ``DiceLoss``、``MultiClassDiceLoss``
     - 需要自定义损失。保持原平方项分母、smooth、激活、类别权重、忽略类别及类别归约规则。
       常见 Dice 公式未必等价；不提供未经核验的一对一替换。
   * - ``FocalLoss``
     - 需要自定义损失，保持原 softmax 分类语义、标签布局、gamma、权重和 reduction。
       sigmoid 二分类 focal loss 不能直接视为等价替代。
   * - ``Pad``
     - 零值 CONSTANT 模式可使用 ``ops.pad(x, padding, mode="constant", value=0)``。
       将原来按首维到末维排列的填充对倒序，再逐对展开；每对内部的前后顺序不变。
       REFLECT 仅在目标设备和输入维度支持时使用 reflect 模式。该接口不支持 SYMMETRIC；
       不要用 reflect 替代，因为两者对边界元素的处理不同。迁移后核对边界值和梯度。
   * - ``LRN``
     - 如需保留局部响应归一化，可在自定义层中按原公式实现。保持通道窗口半径、
       边界处理以及 bias、alpha、beta 的含义；用原模型对照验证前向结果和梯度。
   * - ``FractionalMaxPool3d``
     - 可调用 ``ops.fractional_max_pool3d``，显式传递原 kernel_size、output_size/output_ratio、
       return_indices 和 _random_samples。保持随机样本及返回索引约定；FractionalMaxPool2d 保留。

这里列出的底层函数仅说明本次清理后的可用路径，不代表其未来日落决策已取消。
迁移后应验证前向数值、反向梯度、训练更新、模型加载和目标设备行为。
