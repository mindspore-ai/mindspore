.. py:class:: mindspore.nn.transformer.MoEConfig(expert_num=1, capacity_factor=1.1, aux_loss_factor=0.05, num_experts_chosen=1, expert_group_size=None, group_wise_a2a=True, comp_comm_parallel=False, comp_comm_parallel_degree=2)

    MoE (Mixture of Expert)的配置。

    参数：
        - **expert_num** (int) - 表示使用的专家数量。默认值：1。
        - **capacity_factor** (float) - 表示专家处理的容量关系，其值大于等于1.0。默认值：1.1。
        - **aux_loss_factor** (float) - 表示负载均衡损失（由路由器产生）的平衡系数。相乘的结果会加到总损失函数中。此系数的值小于1.0。默认值：0.05。
        - **num_experts_chosen** (int) - 表示每个标识选择的专家数量，其值小于等于专家数量。默认值：1。
        - **expert_group_size** (int) - 表示每个数据并行组收到的词语（token）数量。默认值：None。该参数只在自动并行且非策略传播模式下起作用。
        - **group_wise_a2a** (bool) - 表示否是使能group-wise alltoall通信，group-wise alltoall通信可以把部分节点间通信转化为节点内通信从而减低通信时间。默认值：False。该参数只有在模型并行数大于1且数据并行数等于专家并行数生效。
        - **comp_comm_parallel** (bool) - 是否使能MoE计算和通信并行，可以通过拆分重叠计算和通信来减少纯通信时间。默认值：False。
        - **comp_comm_parallel_degree** (bool) - 计算和通信的拆分数量。数字越大重叠越多，但会消耗更多的显存。默认值：2。该参数只在comp_comm_parallel为True下生效。
