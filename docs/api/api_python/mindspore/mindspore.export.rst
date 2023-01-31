mindspore.export
================

.. py:function:: mindspore.export(net, *inputs, file_name, file_format, **kwargs)

    将MindSpore网络模型导出为指定格式的文件。

    .. note::
        - 当导出文件格式为AIR、ONNX时，单个Tensor的大小不能超过2GB。
        - 当 `file_name` 没有后缀时，系统会根据 `file_format` 自动添加后缀。
        - 现已支持将 `jit` 修饰的函数导出成MINDIR格式文件。
        - 当导出 `jit` 修饰的函数时，函数内不能包含有类属性参与的计算。

    参数：
        - **net** (Union[Cell, function]) - MindSpore网络结构。
        - **inputs** (Union[Tensor, Dataset, List, Tuple, Number, Bool]) - 网络的输入，如果网络有多个输入，需要一同传入。当传入的类型为 `Dataset` 时，将会把数据预处理行为同步保存起来。需要手动调整batch的大小，当前仅支持获取 `Dataset` 的 `image` 列。
        - **file_name** (str) - 导出模型的文件名称。
        - **file_format** (str) - MindSpore目前支持导出"AIR"，"ONNX"和"MINDIR"格式的模型。

          - **AIR** - Ascend Intermediate Representation。一种Ascend模型的中间表示格式。推荐的输出文件后缀是".air"。
          - **ONNX** - Open Neural Network eXchange。一种针对机器学习所设计的开放式的文件格式。推荐的输出文件后缀是".onnx"。
          - **MINDIR** - MindSpore Native Intermediate Representation for Anf。一种MindSpore模型的中间表示格式。推荐的输出文件后缀是".mindir"。

        - **kwargs** (dict) - 配置选项字典。

          - **quant_mode** (str) - 如果网络是量化感知训练网络，那么 `quant_mode` 需要设置为"QUANT"，否则 `quant_mode` 需要设置为"NONQUANT"。
          - **mean** (float) - 预处理后输入数据的平均值，用于量化网络的第一层。默认值：127.5。
          - **std_dev** (float) - 预处理后输入数据的方差，用于量化网络的第一层。默认值：127.5。
          - **enc_key** (str) - 用于加密的字节类型密钥，有效长度为16、24或者32。
          - **enc_mode** (Union[str, function]) - 指定加密模式，当设置 `enc_key` 时启用。

            - 对于 'AIR'和 'ONNX'格式的模型，当前仅支持自定义加密导出。
            - 对于 'MINDIR'格式的模型，支持的加密选项有： 'AES-GCM'， 'AES-CBC'， 'SM4-CBC'和用户自定义加密算法。默认值："AES-GCM"。
            - 关于使用自定义加密导出的详情，请查看 `教程 <https://www.mindspore.cn/mindarmour/docs/zh-CN/r1.9/model_encrypt_protection.html>`_。
          
          - **dataset** (Dataset) - 指定数据集的预处理方法，用于将数据集的预处理导入MindIR。

          - **obf_config** (dict) - 模型混淆配置选项字典。

            - **type** (str) - 混淆类型，目前支持动态混淆，即 'dynamic' 。
            - **obf_ratio** (Union[str, float]) - 全模型算子的混淆比例，可取浮点数(0, 1]或者字符串 "small" 、 "medium" 、 "large" 。
            - **customized_func** (function) - 在自定义函数模式下需要设置的Python函数，用来控制混淆结构中的选择分支走向。它的返回值需要是bool类型，且是恒定的，用户可以参考不透明谓词进行设置。如果设置了 `customized_func` ，那么在使用 `load` 接口导入模型的时候，需要把这个函数也传入。
            - **obf_password** (int) - 秘密口令，用于password模式，是一个大于0、小于等于int_64_max(9223372036854775807)的整数。如果用户设置了 `obf_password` ，那么在部署混淆模型的时候，需要在调用 :class:`mindspore.nn.GraphCell` 接口中传入 `obf_password` 。需要注意的是，如果用户同时设置了 `customized_func` 和 `obf_password` ，那么password模式将会被采用。
