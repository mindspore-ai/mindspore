Federated-Server
======================

.. py:function:: mindspore.set_fl_context(**kwargs)

    设置联邦学习训练模式的context。

    .. note::
        设置属性时，必须输入属性名称。

    在特定角色的上需要设置不同配置，详细信息参见下表：

    +-------------------------+---------------------------------+-----------------------------+
    | 功能分类                |  配置参数                       |  联邦学习角色               |
    +=========================+=================================+=============================+
    | 组网配置                |  enable_fl                      | scheduler/server/worker     |
    |                         +---------------------------------+-----------------------------+
    |                         |  server_mode                    |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  ms_role                        |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  worker_num                     |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  server_num                     |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  scheduler_ip                   |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  scheduler_port                 |  scheduler/server/worker    |
    |                         +---------------------------------+-----------------------------+
    |                         |  fl_server_port                 |  scheduler/server           |
    |                         +---------------------------------+-----------------------------+
    |                         |  scheduler_manage_port          |  scheduler                  |
    +-------------------------+---------------------------------+-----------------------------+
    | 训练配置                |  start_fl_job_threshold         |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  start_fl_job_time_window       |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  update_model_ratio             |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  update_model_time_window       |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  fl_iteration_num               |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  client_epoch_num               |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  client_batch_size              |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  client_learning_rate           |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  worker_step_num_per_iteration  |  worker                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  global_iteration_time_window   |  server                     |
    +-------------------------+---------------------------------+-----------------------------+
    | 加密配置                |  encrypt_type                   |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  share_secrets_ratio            |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  cipher_time_window             |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  reconstruct_secrets_threshold  |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  dp_eps                         |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  dp_delta                       |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  dp_norm_clip                   |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  sign_k                         |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  sign_eps                       |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  sign_thr_ratio                 |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  sign_global_lr                 |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  sign_dim_out                   |  server                     |
    +-------------------------+---------------------------------+-----------------------------+
    | 认证配置                |  enable_ssl                     |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  client_password                |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  server_password                |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  root_first_ca_path             |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  pki_verify                     |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  root_second_ca_path            |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  equip_crl_path                 |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  replay_attack_time_diff        |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  http_url_prefix                |  server                     |
    +-------------------------+---------------------------------+-----------------------------+
    | 容灾和运维配置          |  fl_name                        |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  config_file_path               |  scheduler/server           |
    |                         +---------------------------------+-----------------------------+
    |                         |  checkpoint_dir                 |  server                     |
    +-------------------------+---------------------------------+-----------------------------+
    | 压缩配置                |  upload_compress_type           |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  upload_sparse_rate             |  server                     |
    |                         +---------------------------------+-----------------------------+
    |                         |  download_compress_type         |  server                     |
    +-------------------------+---------------------------------+-----------------------------+

    参数：
        - **enable_fl** (bool) - 是否启用联邦学习训练模式。默认值：False。
        - **server_mode** (str) - 描述服务器模式，它必须是'FEDERATED_LEARNING'和'HYBRID_TRAINING'中的一个。
        - **ms_role** (str) - 进程在联邦学习模式中的角色，
          必须是'MS_SERVER'、'MS_WORKER'和'MS_SCHED'中的一个。
        - **worker_num** (int) - 云侧训练进程的数量。
        - **server_num** (int) - 联邦学习服务器的数量。默认值：0。
        - **scheduler_ip** (str) - 调度器IP。默认值：'0.0.0.0'。
        - **scheduler_port** (int) - 调度器端口。默认值：6667。
        - **fl_server_port** (int) - 服务器端口。默认值：6668。
        - **start_fl_job_threshold** (int) - 开启联邦学习作业的阈值计数。默认值：1。
        - **start_fl_job_time_window** (int) - 开启联邦学习作业的时间窗口持续时间，以毫秒为单位。默认值：300000。
        - **update_model_ratio** (float) - 计算更新模型阈值计数的比率。默认值：1.0。
        - **update_model_time_window** (int) - 更新模型的时间窗口持续时间，以毫秒为单位。默认值：300000。
        - **fl_name** (str) - 联邦学习作业名称。默认值：""。
        - **fl_iteration_num** (int) - 联邦学习的迭代次数，即客户端和服务器的交互次数。默认值：20。
        - **client_epoch_num** (int) - 客户端训练epoch数量。默认值：25。
        - **client_batch_size** (int) - 客户端训练数据batch数。默认值：32。
        - **client_learning_rate** (float) - 客户端训练学习率。默认值：0.001。
        - **worker_step_num_per_iteration** (int) - 端云联邦中，云侧训练进程在与服务器通信之前的独立训练步数。默认值：65。
        - **encrypt_type** (str) - 用于联邦学习的安全策略，可以是'NOT_ENCRYPT'、'DP_ENCRYPT'、
          'PW_ENCRYPT'、'STABLE_PW_ENCRYPT'或'SIGNDS'。如果是'DP_ENCRYPT'，则将对客户端应用差分隐私模式，
          隐私保护效果将由上面所述的dp_eps、dp_delta、dp_norm_clip确定。如果'PW_ENCRYPT'，则将应用成对（pairwise，PW）安全聚合
          来保护客户端模型在跨设备场景中不被窃取。如果'STABLE_PW_ENCRYPT'，则将应用成对安全聚合来保护客户端模型在云云联邦场景中
          免受窃取。如果'SIGNDS'，则将在于客户端上使用SignDS策略。SignDS的介绍可以参照：
          `SignDS-FL: Local Differentially Private Federated Learning with Sign-based Dimension Selection <https://dl.acm.org/doi/abs/10.1145/3517820>`_。
          默认值：'NOT_ENCRYPT'。
        - **share_secrets_ratio** (float) - PW：参与秘密分享的客户端比例。默认值：1.0。
        - **cipher_time_window** (int) - PW：每个加密轮次的时间窗口持续时间，以毫秒为单位。默认值：300000。
        - **reconstruct_secrets_threshold** (int) - PW：秘密重建的阈值。默认值：2000。
        - **dp_eps** (float) - DP：差分隐私机制的epsilon预算。dp_eps越小，隐私保护效果越好。默认值：50.0。
        - **dp_delta** (float) - DP：差分隐私机制的delta预算，通常等于客户端数量的倒数。dp_delta越小，隐私保护效果越好。默认值：0.01。
        - **dp_norm_clip** (float) - DP：差分隐私梯度裁剪的控制因子。建议其值为0.5~2。默认值：1.0。
        - **sign_k** (float) - SignDS：Top-k比率，即Top-k维度的数量除以维度总数。建议取值范围在(0, 0.25]内。默认值：0.01。
        - **sign_eps** (float) - SignDS：隐私预算。该值越小隐私保护力度越大，精度越低。建议取值范围在(0, 100]内。默认值：100。
        - **sign_thr_ratio** (float) - SignDS：预期Top-k维度的阈值。建议取值范围在[0.5, 1]内。默认值：0.6。
        - **sign_global_lr** (float) - SignDS：分配给选定维的常量值。适度增大该值会提高收敛速度，但有可能让模型梯度爆炸。取值必须大于0。默认值：1。
        - **sign_dim_out** (int) - SignDS：输出维度的数量。建议取值范围在[0, 50]内。默认值：0。
        - **config_file_path** (str) - 用于集群容灾恢复的配置文件路径、认证相关参数以及文件路径、评价指标文件路径和运维相关文件路径。默认值：""。
        - **scheduler_manage_port** (int) - 用于扩容/缩容的调度器管理端口。默认值：11202。
        - **enable_ssl** (bool) - 设置联邦学习开启SSL安全通信。默认值：False。
        - **client_password** (str) - 解密客户端证书中存储的秘钥的密码。默认值：""。
        - **server_password** (str) - 解密服务器证书中存储的秘钥的密码。默认值：""。
        - **pki_verify** (bool) - 如果为True，则将打开服务器和客户端之间的身份验证。
          还应从https://pki.consumer.huawei.com/ca/下载Root CA证书、Root CA G2证书和移动设备CRL证书。
          需要注意的是，只有当客户端是具有HUKS服务的Android环境时，pki_verify可以为True。默认值：False。
        - **root_first_ca_path** (str) - Root CA证书的文件路径。当pki_verify为True时，需要设置该值。默认值：""。
        - **root_second_ca_path** (str) - Root CA G2证书的文件路径。当pki_verify为True时，需要设置该值。默认值：""。
        - **equip_crl_path** (str) - 移动设备CRL证书的文件路径。当pki_verify为True时，需要设置该值。默认值：""。
        - **replay_attack_time_diff** (int) - 证书时间戳验证的最大可容忍错误（毫秒）。默认值：600000。
        - **http_url_prefix** (str) - 设置联邦学习端云通信的http路径。默认值：""。
        - **global_iteration_time_window** (int) - 一次迭代的全局时间窗口，轮次（ms）。默认值：3600000。
        - **checkpoint_dir** (str) - server读取和保存模型文件的目录。若没有设置则不读取和保存模型文件。默认值：""。
        - **upload_compress_type** (str) - 上传压缩方法。可以是'NO_COMPRESS'或'DIFF_SPARSE_QUANT'。如果是'NO_COMPRESS'，则不对上传的模型
          进行压缩。如果是'DIFF_SPARSE_QUANT'，则对上传的模型使用权重差+稀疏+量化压缩策略。默认值：'NO_COMPRESS'。
        - **upload_sparse_rate** (float) - 上传压缩稀疏率。稀疏率越大，则压缩率越小。取值范围：(0, 1.0]。默认值：0.4。
        - **download_compress_type** (str) - 下载压缩方法。可以是'NO_COMPRESS'或'QUANT'。如果是'NO_COMPRESS'，则不对下载的模型进行压缩。
          如果是'QUANT'，则对下载的模型使用量化压缩策略。默认值：'NO_COMPRESS'。

    异常：
        - **ValueError** - 如果输入key不是联邦学习模式context中的属性。


.. py:function:: mindspore.get_fl_context(attr_key)

    根据key获取联邦学习模式context中的属性值。

    参数：
        - **attr_key** (str) - 属性的key。
          请参考 `set_fl_context` 中的参数来决定应传递的key。

    返回：
        根据key返回属性值。

    异常：
        - **ValueError** - 如果输入key不是联邦学习模式context中的属性。

