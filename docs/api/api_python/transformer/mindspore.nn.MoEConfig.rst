.. py:class:: mindspore.nn.transformer.MoEConfig(expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1)

    MoE (Mixture of Expert)的配置。

    **参数：**

    - **expert_num** (int) - 表示使用的专家数量。默认值：1。
    - **capacity_factor** (float) - 表示专家处理的容量关系，其值大于等于1.0。默认值：1.1。
    - **aux_loss_factor** (float) - 表示负载均衡损失（由路由器产生）的平衡系数。相乘的结果会加到总损失函数中。此系数的值小于1.0。默认值：0.05。
    - **num_experts_chosen** (int) - 表示每个标识选择的专家数量。默认值：1。
