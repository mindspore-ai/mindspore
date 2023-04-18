mindspore.get_ps_context
=========================

.. py:function:: mindspore.get_ps_context(attr_key)

    根据key获取参数服务器训练模式上下文中的属性值。

    参数：
        - **attr_key** (str) - 属性的key。

          - enable_ps (bool)：表示是否启用参数服务器训练模式。默认值： ``False`` 。
          - config_file_path (string)：配置文件路径，用于容灾恢复等。默认值： ``''`` 。
          - scheduler_manage_port (int)：调度器HTTP端口，对外开放用于接收和处理用户扩容/缩容等请求。默认值： ``11202`` 。
          - enable_ssl (bool)：设置是否打开SSL认证。默认值： ``False`` 。
          - client_password (str)：用于解密客户端证书密钥的密码。默认值： ``''`` 。
          - server_password (str)：用于解密服务端证书密钥的密码。默认值： ``''`` 。

    返回：
        根据key返回属性值。

    异常：
        - **ValueError** - 输入key不是参数服务器训练模式上下文中的属性。
