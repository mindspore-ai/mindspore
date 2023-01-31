mindspore.load
=======================================

.. py:function:: mindspore.load(file_name, **kwargs)

    加载MindIR文件。

    返回一个可以由 `GraphCell` 执行的对象，更多细节参见类 :class:`mindspore.nn.GraphCell` 。

    参数：
        - **file_name** (str) - MindIR文件的全路径名。
        - **kwargs** (dict) - 配置项字典。

          - **dec_key** (bytes) - 用于解密的字节类型密钥。有效长度为 16、24 或 32。
          - **dec_mode** (Union[str, function]) - 指定解密模式，设置dec_key时生效。可选项： 'AES-GCM' | 'SM4-CBC' | 'AES-CBC' ｜ 自定义解密函数。默认值："AES-GCM"。

            - 关于使用自定义解密加载的详情，请查看 `教程 <https://www.mindspore.cn/mindarmour/docs/zh-CN/r1.9/model_encrypt_protection.html>`_。

          - **obf_func** (function) - 导入混淆模型所需要的函数，可以参考 `obfuscate_model() <https://www.mindspore.cn/docs/zh-CN/r2.0.0-alpha/api_python/mindspore/mindspore.obfuscate_model.html>`_ 了解详情。

    返回：
        GraphCell，一个可以由 `GraphCell` 构成的可执行的编译图。

    异常：
        - **ValueError** - MindIR文件名不存在或 `file_name` 不是string类型。
        - **RuntimeError** - 解析MindIR文件失败。
