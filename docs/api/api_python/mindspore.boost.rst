mindspore.boost
==============================

Boost能够自动加速网络，如减少BN/梯度冻结/累积梯度等。

.. note::
    此特性为测试版本，我们仍在改进其功能。

.. py:class:: mindspore.boost.AutoBoost(level="O0", boost_config_dict="")

    MindSpore自动优化算法库。

    参数：
        - **level** (str) - Boost的配置级别，默认值："O0"。

          - "O0"：不变化。
          - "O1"：启用boost模式，性能将提升约20%，准确率保持不变。
          - "O2"：启用boost模式，性能将提升约30%，准确率下降小于3%。

        - **boost_config_dict** (dict) - 用户可配置的超参字典，建议的格式如下：

          .. code-block::

              {
                  "boost": {
                      "mode": "auto",
                      "less_bn": False,
                      "grad_freeze": False,
                      "adasum": False,
                      "grad_accumulation": False,
                      "dim_reduce": False},

                  "common": {
                      "gradient_split_groups": [50, 100],
                      "device_number": 8},

                  "less_bn": {
                      "fn_flag": True,
                      "gc_flag": True},

                  "grad_freeze": {
                      "param_groups": 10,
                      "freeze_type": 1,
                      "freeze_p": 0.7,
                      "total_steps": 65536},

                  "grad_accumulation": {
                      "grad_accumulation_step": 1},

                  "dim_reduce": {
                      "rho": 0.55,
                      "gamma": 0.9,
                      "alpha": 0.001,
                      "sigma": 0.4,
                      "n_components": 32,
                      "pca_mat_path": None,
                      "weight_load_dir": None,
                      "timeout": 1800}

              }

          - boost：

            - mode (str)：Boost配置模式，支持 ["auto", "manual", "enable_all", "disable_all"]。默认值： "auto"。

              - auto：自动配置，取决于Model类中的 `boost_level` 参数配置。
              - manual：在 `boost_config_dict` 中人工配置。
              - enable_all：开启所有boost算法。
              - disable_all：关闭所有boost算法。

            - less_bn (bool)：是否开启LessBN算法，默认：False
            - grad_freeze (bool)：是否开启梯度冻结算法，默认：False。
            - adasum (bool)：是否开启自适应求和算法，默认：False。
            - grad_accumulation (bool)：是否开启梯度累加算法，默认：False。
            - dim_reduce (bool)：是否开启降维训练算法，默认：False。

              如果开启dim_reduce算法，其他算法会失效。
              如果开启grad_freeze算法，同时关闭dim_reduce，其他算法会失效。

          - common：

            - gradient_split_groups (list)：网络的梯度分割点，默认：[50, 100]。
            - device_number (int)：设备数，默认：8。

          - less_bn：

            - fn_flag (bool)：是否采用fn替换fc，默认：替换。
            - gc_flag (bool)：是否启用gc，默认：启用gc。

          - grad_freeze：

            - param_groups (int)：参数分组数量，默认值：10。
            - freeze_type (int)：梯度冻结策略，参数选择[0, 1]，默认值：1。
            - freeze_p (float)：梯度冻结概率，默认值：0.7。
            - total_steps (int)：总训练步数，默认值：65536。

          - grad_accumulation：

            - grad_accumulation_step (int)：累加梯度的步数，默认值：1。

          - dim_reduce：

            dim_reduce主要原理：

            .. math::

                \begin{align}
                grad\_k &= pca\_mat \cdot grad\\
                dk &= - bk \cdot grad\_k\\
                sk &= rho ^ m \cdot dk\\
                delta\_loss &= sigma \cdot grad\_k.T \cdot sk
                \end{align}

            其中：

            - pca_mat (array)：维度(k*n)，k是 *n_components* 的大小，n是权重的大小。
            - bk (array)：维度(k*k)，bk是拟牛顿法中的对称正定矩阵。

            我们需要找到满足以下条件的m：

            .. math::
                new\_loss < old\_loss + delta\_loss

            然后使用 *delta_grad* 去更新模型的权重：

            .. math::

                \begin{align}
                grad\_k\_proj &= pca\_mat.T \cdot grad\_k\\
                new\_grad\_momentum &= gamma \cdot old\_grad\_momentum + grad - grad\_k\_proj\\
                delta\_grad &= alpha \cdot new\_grad\_momentum - pca\_mat.T \cdot sk
                \end{align}

            - rho (float)：超参，一般无需调整，默认值：0.55。
            - gamma (float)：超参，一般无需调整，默认值：0.9。
            - alpha (float)：超参，一般无需调整，默认值：0.001。
            - sigma (float)：超参，一般无需调整，默认值：0.4。
            - n_components (int)：PCA后的维度，默认值：32。
            - pca_mat_path (str)：PCA矩阵的加载路径，使用绝对路径，默认值：None。
            - weight_load_dir (str)：以checkpoint形式保存的权重加载路径，用于计算PCA矩阵，默认值：None。
            - timeout (int)：加载PCA矩阵的最长等待时间，默认值：1800(s)。

          用户可以通过加载JSON文件或者直接使用字典来配置 *boost_config_dict*。
          未配置的参数会使用默认值。

    异常：
        - **ValueError** - Boost的模式不在["auto", "manual", "enable_all", "disable_all"]这个列表中。

    .. py:method:: network_auto_process_eval(network)

        使用Boost算法推理。

        参数：
            - **network** (Cell) - 推理网络。

    .. py:method:: network_auto_process_train(network, optimizer)

        使用Boost算法训练。

        参数：
            - **network** (Cell) - 训练网络。
            - **optimizer** (Cell) - 用于更新权重的优化器。

.. py:class:: mindspore.boost.OptimizerProcess(opt)

    处理Boost的优化器。目前支持给优化器添加梯度中心化和创建新的优化器。

    参数：
        - **opt** (Cell) - 使用的优化器。

    .. py:method:: add_grad_centralization(network)

        添加梯度中心化。

        参数：
            - **network** (Cell) - 训练网络。

    .. py:method:: build_gc_params_group(params_dict, parameters)

        构建梯度中心化的分组权重。

        参数：
            - **params_dict** (dict) - 训练权重的字典。
            - **parameters** (list) - 训练权重的列表。

    .. py:method:: build_params_dict(network)

        构建网络权重的字典。

        参数：
            - **network** (Cell) - 训练网络。

    .. py:method:: generate_new_optimizer()

        生成新的优化器。

.. py:class:: mindspore.boost.ParameterProcess()

    处理Boost网络的权重。当前支持创建分组参数和自动设置网络梯度切分点。

    .. py:method:: assign_parameter_group(parameters, split_point=None)

        设置分组权重。

        参数：
            - **parameters** (list) - 训练网络的权重。
            - **split_point** (list) - 网络梯度切分点。默认为None。

    .. py:method:: generate_group_params(parameters, origin_params)

        创建分组权重。

        参数：
            - **parameters** (list) - 训练网络的新权重。
            - **origin_params** (list) - 训练网络的初始权重。

.. py:class:: mindspore.boost.BoostTrainOneStepCell(network, optimizer, sens=1.0)

    Boost网络训练封装类。

    用优化器封装网络。使用输入训练网络来获取结果。反向图在 *construct* 函数中创建，以更新参数，并且支持多种不同的并行模式。

    参数：
        - **network** (Cell) - 训练网络，当前网络只支持单个输出。
        - **optimizer** (Union[Cell]) - 用于更新网络参数的优化器。
        - **sens** (numbers.Number) - 作为反向传播输入要填充的缩放数，默认值为1.0。

    输入：
        - **\*inputs** (Tuple(Tensor)) - 网络的所有输入组成的元组。

    输出：
        Tuple，包含三个Tensor，分别为损失函数值、溢出状态和当前损失缩放系数。

        - loss(Tensor)，标量Tensor。
        - overflow(Tensor)，标量Tensor，类型为bool。
        - loss scaling value(Tensor)，标量Tensor。

    异常：
        - **TypeError** - 如果 `sens` 不是一个数字。

    .. py:method:: adasum_process(loss, grads)

        使用Adasum算法训练。

        参数：
            - **loss** (Tensor) - 网络训练的loss值。
            - **grads** (tuple(Tensor)) - 网络训练过程中的梯度。

        返回：
            Tensor，网络训练过程中得到的loss值。

    .. py:method:: check_adasum_enable()

        Adasum算法仅在多卡或者多机场景生效，并且要求卡数符合2的n次方，该函数用来判断adasum算法能否生效。

        返回：
            enable_adasum (bool)，Adasum算法是否生效。

    .. py:method:: check_dim_reduce_enable()

        获取当前是否使用降维二阶训练算法训练。

        返回：
            enable_dim_reduce (bool)，降维二阶训练算法是否生效。

    .. py:method:: gradient_accumulation_process(loss, grads, sens, *inputs)

        使用梯度累积算法训练。

        参数：
            - **loss** (Tensor) - 网络训练的loss值。
            - **grads** (tuple(Tensor)) - 网络训练过程中的梯度。
            - **sens** (Tensor) - 作为反向传播输入要填充的缩放数。
            - **inputs** (tuple(Tensor)) - 网络训练的输入。

        返回：
            Tensor，网络训练过程中得到的loss值。

    .. py:method:: gradient_freeze_process(*inputs)

        使用梯度冻结算法训练。

        参数：
            - **inputs** (tuple(Tensor)) - 网络训练的输入。

        返回：
            Tensor，网络训练过程中得到的loss值。

.. py:class:: mindspore.boost.BoostTrainOneStepWithLossScaleCell(network, optimizer, scale_sense)

    使用混合精度功能的Boost训练网络。

    实现了包含损失缩放（loss scale）的单次训练。它使用网络、优化器和用于更新损失缩放系数（loss scale）的Cell(或一个Tensor)作为参数。可在host侧或device侧更新损失缩放系数。BoostTrainOneStepWithLossScaleCell会被编译成图，其中inputs作为输入数据。张量类型参数 `scale_sense` 作为损失缩放时使用的值。
    如果需要在host侧更新，使用Tensor作为 `scale_sense` 。如果需要在device侧更新，使用可更新损失缩放系数的Cell实例作为 `scale_sense` 。

    参数：
        - **network** (Cell) - 训练网络，当前网络只支持单个输出。
        - **optimizer** (Cell) - 用于更新网络参数的优化器。
        - **scale_sense** (Union[Tensor, Cell]) - 如果此值为Cell类型，`BoostTrainOneStepWithLossScaleCell` 会调用它来更新损失缩放系数。如果此值为Tensor类型，可调用 `set_sense_scale` 来更新损失缩放系数，shape为 :math:`()` 或 :math:`(1,)` 。

    输入：
        - **\*inputs** (Tuple(Tensor)) - 网络的所有输入组成的元组。

    输出：
        Tuple，包含三个Tensor，分别为损失函数值、溢出状态和当前损失缩放系数。

        - **loss** (Tensor) - 标量Tensor。
        - **overflow** (Tensor) - 标量Tensor，类型为bool。
        - **loss scaling value** (Tensor) - 标量Tensor。

    异常：
        - **TypeError** - `scale_sense` 既不是Cell，也不是Tensor。
        - **ValueError** - `scale_sense` 的shape既不是(1,)也不是()。

.. py:class:: mindspore.boost.LessBN(network, fn_flag=False)

    LessBN算法，可以在不损失网络精度的前提下，自动减少网络中批归一化（Batch Normalization）的数量，来提升网络性能。

    参数：
        - **network** (Cell) - 待训练的网络模型。
        - **fn_flag** (bool) - 是否将网络中最后一个全连接层替换为全归一化层。默认值：False。

.. py:class:: mindspore.boost.GradientFreeze(param_groups, freeze_type, freeze_p, total_steps)

    梯度冻结算法，根据指定策略随机冻结某些层的梯度，来提升网络训练性能。
    冻结的层数和冻结的概率均可由用户配置。

    参数：
        - **param_groups** (Union[tuple, list]) - 梯度冻结训练的权重。
        - **freeze_type** (int) - 梯度冻结训练的策略。
        - **freeze_p** (float) - 梯度冻结训练的概率。
        - **total_steps** (int) - 整个训练过程的总的步数。

    .. py:method:: freeze_generate(network, optimizer)

        生成梯度冻结的网络与优化器。

        参数：
            - **network** (Cell) - 训练网络。
            - **optimizer** (Cell) - 用于更新权重的优化器。

    .. py:method:: generate_freeze_index_sequence(parameter_groups_number, freeze_strategy, freeze_p, total_steps)

        生成梯度冻结每一步需要冻结的层数。

        参数：
            - **parameter_groups_number** (int) - 梯度冻结训练的权重个数。
            - **freeze_strategy** (int) - 梯度冻结训练的策略。
            - **freeze_p** (float) - 梯度冻结训练的概率。
            - **total_steps** (int) - 整个训练过程的总的步数。

    .. py:method:: split_parameters_groups(net, freeze_para_groups_number)

        拆分用于梯度冻结训练的权重。

        参数：
            - **net** (Cell) - 训练网络。
            - **freeze_para_groups_number** (int) - 梯度冻结训练的权重个数。

.. py:class:: mindspore.boost.FreezeOpt(opt, train_parameter_groups=None, train_strategy=None)

    支持梯度冻结训练的优化器。

    参数：
        - **opt** (Cell) - 非冻结优化器实例，如 *Momentum*，*SGD*。
        - **train_parameter_groups** (Union[tuple, list]) - 梯度冻结训练的权重。
        - **train_strategy** (Union[tuple(int), list(int), Tensor]) - 梯度冻结训练的策略。

.. py:function:: mindspore.boost.freeze_cell(reducer_flag, network, optimizer, sens, grad, use_grad_accumulation, mean=None, degree=None, max_accumulation_step=1)

    提供带梯度冻结的网络Cell。

    参数：
        - **reducer_flag** (bool) - 是否分布式训练。
        - **network** (Cell) - 训练网络。
        - **optimizer** (Cell) - 优化器。
        - **sens** (numbers.Number) - 损失缩放系数。
        - **grad** (tuple(Tensor)) - 网络梯度。
        - **use_grad_accumulation** (bool) - 是否使用梯度累积。
        - **mean** (bool) - 可选参数，梯度是否求平均，仅分布式训练时生效。默认值为None。
        - **degree** (int) - 可选参数，device卡数，仅分布式训练时生效。默认值为None。
        - **max_accumulation_step** (int) - 可选参数，梯度累积步数。默认值为1。

.. py:class:: mindspore.boost.GradientAccumulation(max_accumulation_step, optimizer)

    梯度累积算法，在累积多个step的梯度之后，再用来更新网络权重，可以提高训练效率。

    参数：
        - **max_accumulation_step** (int) - 累积梯度的步数。
        - **optimizer** (Cell) - 网络训练使用的优化器。

.. py:class:: mindspore.boost.AdaSum(rank, device_number, group_number, parameter_tuple)

    Adaptive Summation(AdaSum)是一种优化深度学习模型并行训练的算法，它可以提升不同规模集群训练的精度，减小不同规模集群调参难度。

    参数：
        - **rank** (int) - 总的训练的卡数。
        - **device_number** (int) - 单机的卡数。
        - **group_number** (int) - 分组的数量。
        - **parameter_tuple** (Tuple(Parameter)) - 网络训练权重组成的元组。

    输入：
        - **delta_weights** (Tuple(Tensor)) - 梯度tuple。
        - **parameters** (Tuple(Parameter)) - 当前权重组成的元组。
        - **old_parameters** (Tuple(Parameter)) - 旧的权重组成的元组。

    输出：
        - **adasum_parameters** (Tuple(Tensor)) - adasum处理后更新的权重。

.. py:class:: mindspore.boost.DimReduce(network, optimizer, weight, pca_mat_local, n_components, rho, gamma, alpha, sigma, rank, rank_size)

    降维训练(dimension reduce training)是一种优化深度学习模型训练的算法，它可以加速模型的收敛。

    算法主要原理：

    .. math::

        \begin{align}
        grad\_k &= pca\_mat \cdot grad\\
        dk &= - bk \cdot grad\_k\\
        sk &= rho ^ m \cdot dk\\
        delta\_loss &= sigma \cdot grad\_k.T \cdot sk
        \end{align}

    其中:

    - pca_mat (array): PCA矩阵，维度(k*n)，k是 `n_components` 的大小，n是权重的大小。
    - bk (array): 维度(k*k)，bk是拟牛顿法中的对称正定矩阵。

    我们需要找到满足以下条件的m:

    .. math::
        new\_loss < old\_loss + delta\_loss

    然后使用delta_grad去更新模型的权重:

    .. math::

        \begin{align}
        grad\_k\_proj &= pca\_mat.T \cdot grad\_k\\
        new\_grad\_momentum &= gamma \cdot old\_grad\_momentum + grad - grad\_k\_proj\\
        delta\_grad &= alpha \cdot new\_grad\_momentum - pca\_mat.T \cdot sk
        \end{align}

    参数：
        - **network** (Cell) - 训练网络，只支持单输出。
        - **optimizer** (Union[Cell]) - 更新权重的优化器。
        - **weight** (Tuple(Parameter)) - 网络权重组成的元组。
        - **pca_mat_local** (numpy.ndarray) - 用于PCA操作的，经过切分的PCA转换矩阵，维度为k*n，k是切分的 `n_components` 的大小，n是权重的大小。
        - **n_components** (int) - PCA的主成分维度(components)。
        - **rho** (float) - 超参。
        - **gamma** (float) - 超参。
        - **alpha** (float) - 超参。
        - **sigma** (float) - 超参。
        - **rank** (int) - Rank编号。
        - **rank_size** (int) - Rank总数。

    输入：
        - **loss** (Tensor) - 网络loss，标量Tensor。
        - **old_grad** (Tuple(Tensor)) - 网络权重提取组成的元组。
        - **weight** (Tuple(Tensor)) - 网络权重组成的元组。
        - **weight_clone** (Tuple(Tensor)) - 网络权重的副本。
        - **\*inputs** (Tuple(Tensor)) - 网络的所有输入组成的元组。

    输出：
        - **loss** (Tensor) - 网络loss，标量Tensor。

.. py:class:: mindspore.boost.GroupLossScaleManager(init_loss_scale, loss_scale_groups)

    增强型混合精度算法支持不同loss scale的多层应用和损失尺度的动态更新。

    参数：
        - **init_loss_scale** (Number) - 初始化loss scale。
        - **loss_scale_groups** (List) - 从参数列表里分离出来的loss scale组。

    输入：
        - **x** (Tensor) - 最后一个operator的输出。
        - **layer1** (int) - 当前网络层的值。
        - **layer2** (int) - 最后一个网络层的值。

    输出：
        - **x** (Tensor) - _DynamicLossScale operator的输出。

    .. py:method:: get_loss_scale()

        获取loss scale的值。

        返回：
            bool，`loss_scale` 的值。

    .. py:method:: get_update_cell()

        返回 :class:`mindspore.boost.GroupLossScaleManager` 实例。

        返回：
            :class:`mindspore.boost.GroupLossScaleManager` 实例。

    .. py:method:: set_loss_scale_status(loss_scale_number, init_loss_scale)

        生成动态loss scale元组，并设置溢出状态列表。

        参数：
            - **loss_scale_number** (int) - loss scale的数量。
            - **init_loss_scale** (float) - 已初始化的loss scale。

    .. py:method:: update_loss_scale_status(layer, update_ratio)

        更新动态loss scale。

        参数：
            - **layer** (int) - 当前层。
            - **update_ratio** (float) - 更新loss scale的当前比例。

        输出：
            float，新loss scale的值。

.. automodule:: mindspore.boost
    :members:
