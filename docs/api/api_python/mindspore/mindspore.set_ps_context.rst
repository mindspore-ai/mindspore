mindspore.set_ps_context
=========================

.. py:function:: mindspore.set_ps_context(**kwargs)

    设置参数服务器训练模式的上下文。

    .. note::
        参数服务器训练模式只在图模式下支持。
        需要给参数服务器训练模式设置其他的环境变量。这些环境变量如下所示：

        - MS_SERVER_NUM：表示参数服务器数量。
        - MS_WORKER_NUM：表示工作进程数量。
        - MS_SCHED_HOST：表示调度器IP地址。
        - MS_SCHED_PORT：表示调度器开启的监听端口。
        - MS_ROLE：表示进程角色，角色列表如下：

          - MS_SCHED：表示调度器。
          - MS_WORKER：表示工作进程。
          - MS_PSERVER/MS_SERVER：表示参数服务器。

    参数：
        - **enable_ps** (bool) - 表示是否启用参数服务器训练模式。只有在enable_ps设置为True后，环境变量才会生效。默认值：False。
        - **config_file_path** (string) - 配置文件路径，用于容灾恢复等, 目前参数服务器训练模式仅支持Server容灾。默认值：''。
        - **scheduler_manage_port** (int) - 调度器HTTP端口，对外开放用于接收和处理用户扩容/缩容等请求。默认值：11202。
        - **enable_ssl** (bool) - 设置是否打开SSL认证。默认值：True。
        - **client_password** (str) - 用于解密客户端证书密钥的密码。默认值：''。
        - **server_password** (str) - 用于解密服务端证书密钥的密码。默认值：''。


    异常：
        - **ValueError** - 输入key不是参数服务器训练模式上下文中的属性。
