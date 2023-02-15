mindspore.obfuscate_model
=========================

.. py:function:: mindspore.obfuscate_model(obf_config, **kwargs)

    对MindIR格式的模型进行混淆，混淆主要是修改模型的网络结构但不影响它的推理精度，混淆后的模型可以防止被盗用。

    参数：
        - **obf_config** (dict) - 模型混淆配置选项字典。

          - **type** (str) - 混淆类型，目前支持动态混淆，即 'dynamic'。
          - **original_model_path** (str) - 待混淆的MindIR模型地址。如果该模型是加密文件的，则需要在 `kwargs` 中传入 `enc_key` 和 `enc_mode` 。
          - **save_model_path** (str) - 混淆模型的保存地址。
          - **model_inputs** (list[Tensor]) - 模型的推理输入，Tensor的值可以是随机的，和使用 :func:`mindspore.export` 接口类似。
          - **obf_ratio** (Union[str, float]) - 全模型算子的混淆比例，可取浮点数(0, 1]或者字符串"small"、"medium"、"large"。
          - **customized_func** (function) - 在自定义函数模式下需要设置的Python函数，用来控制混淆结构中的选择分支走向。它的返回值需要是bool类型，且是恒定的，用户可以参考不透明谓词进行设置。如果设置了 `customized_func` ，那么在使用 :func:`mindspore.load` 接口导入模型的时候，需要把这个函数也传入。
          - **obf_random_seed** (int) - 混淆随机种子，用于确定混淆分支的分布和权重混淆系数，是一个取值范围为(0, 9223372036854775807]的整数。如果用户设置了 `obf_random_seed` ，那么在部署混淆模型的时候，需要在 :class:`mindspore.nn.GraphCell` 接口中传入 `obf_random_seed` 。需要注意的是，如果用户同时设置了 `customized_func` 和 `obf_random_seed` ，那么后一种模式将会被采用。

        - **kwargs** (dict) - 配置选项字典。

          - **enc_key** (bytes) - 用于加密的字节类型密钥，有效长度为16、24或者32。
          - **enc_mode** (str) - 指定加密模式，当设置 `enc_key` 时启用。支持的加密选项有： 'AES-GCM'， 'AES-CBC', 'SM4-CBC'。默认值："AES-GCM"。

    异常：
        - **TypeError** - `obf_config` 不是字典类型。
        - **ValueError** - 传入了 `enc_key` 但是 `enc_mode`　不在["AES-GCM", "AES-CBC", "SM4-CBC"]内。
        - **ValueError** - `obf_config` 没有提供 `original_model_path` 。
        - **ValueError** - `original_model_path` 中的模型是已经混淆过的。
        - **ValueError** - `obf_config` 没有提供 `save_model_path` 。
        - **ValueError** - `obf_config` 没有提供 `obf_ratio` 。
        - **ValueError** - `customized_func` 和 `obf_random_seed` 都不在 `obf_config` 里面。
        - **ValueError** - `obf_random_seed` 的取值没有在(0, 9223372036854775807]内。
        - **ValueError** - `file_path` 不存在或者不以 '.mindir' 结尾。
