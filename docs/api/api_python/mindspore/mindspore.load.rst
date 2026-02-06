mindspore.load
=======================================

.. py:function:: mindspore.load(file_name, **kwargs)

    加载MindIR文件。

    返回一个可以由 `GraphCell` 执行的对象，更多细节参见类 :class:`mindspore.nn.GraphCell` 。

    参数：
        - **file_name** (str) - MindIR文件的全路径名。
        - **kwargs** (dict) - 配置项字典。

          - **dec_key** (bytes) - 用于解密的字节类型密钥。有效长度为 16、24 或 32。
          - **dec_mode** (Union[str, function]，可选) - 指定解密模式，设置 `dec_key` 时生效。

            - 可选项： ``'AES-GCM'``、 ``'SM4-CBC'`` 、 ``'AES-CBC'`` 或自定义解密函数。默认值： ``'AES-GCM'`` 。

            - 关于使用自定义解密加载的详情，请查看 `教程 <https://www.mindspore.cn/mindarmour/docs/zh-CN/master/model_encrypt_protection.html>`_。

    返回：
        GraphCell，一个可以由 `GraphCell` 构成的可执行的编译图。

    异常：
        - **NotImplementedError** - 动态结构混淆已不再支持。
        - **ValueError** - MindIR文件名不存在或 `file_name` 不是string类型。
        - **RuntimeError** - 解析MindIR文件失败。

    教程样例：
        - `保存与加载 - 保存和加载MindIR
          <https://mindspore.cn/tutorials/zh-CN/master/beginner/save_load.html#保存和加载mindir>`_