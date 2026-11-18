mindspore.context.get_ps_context
================================

.. py:function:: mindspore.context.get_ps_context(attr_key)

    根据key获取参数服务器训练模式上下文中的属性值，此接口将在后续版本中废弃。

    参数：
        - **attr_key** (str) - 属性的key。

          - enable_ps (bool，可选)：表示是否启用参数服务器训练模式。默认值： ``False`` 。
          - config_file_path (str，可选)：配置文件路径，用于容灾恢复等。默认值： ``''`` 。
          - enable_ssl (bool，可选)：设置是否打开SSL认证。默认值： ``False`` 。关闭时需要用户审视并确认分布式任务所在网络环境的安全性。

    返回：
        根据key返回属性值。

    异常：
        - **ValueError** - 输入key不是参数服务器训练模式上下文中的属性。
