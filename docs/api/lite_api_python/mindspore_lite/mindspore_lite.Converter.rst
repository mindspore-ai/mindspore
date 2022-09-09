mindspore_lite.Converter
========================

.. py:class:: mindspore_lite.Converter(fmk_type, model_file, output_file, weight_file="", config_file="", weight_fp16=False, input_shape=None, input_format=Format.NHWC, input_data_type=DataType.FLOAT32, output_data_type=DataType.FLOAT32, export_mindir=ModelType.MINDIR_LITE, decrypt_key="", decrypt_mode="AES-GCM", enable_encryption=False, encrypt_key="", infer=False, train_model=False, no_fusion=False)

    转换用于转换第三方模型。

    .. note::
        参数默认值是None时表示不设置。

    参数：
        - **fmk_type** (FmkType) - 输入模型框架类型。选项：FmkType.TF | FmkType.CAFFE | FmkType.ONNX | FmkType.MINDIR | FmkType.TFLITE。
        - **model_file** (str) - 输入模型文件路径。e.g. "/home/user/model.prototxt"。选项：TF: "\*.pb" | CAFFE: "\*.prototxt" | ONNX: "\*.onnx" | MINDIR: "\*.mindir" | TFLITE: "\*.tflite"。
        - **output_file** (str) - 输出模型文件路径。可自动生成.ms后缀。e.g. "/home/user/model.prototxt"，它将生成名为model.prototxt.ms的模型在/home/user/路径下。
        - **weight_file** (str，可选) - 输入模型权重文件。仅当输入模型框架类型为FmkType.CAFFE时必选。e.g. "/home/user/model.caffemodel"。默认值：""。
        - **config_file** (str，可选) - 作为训练后量化或离线拆分算子并行的配置文件路径，禁用算子融合功能并将插件设置为so路径。默认值：""。
        - **weight_fp16** (bool，可选) - 在Float16数据类型中序列化常量张量，仅对Float32数据类型中的常量张量有效。默认值：""。
        - **input_shape** (dict{str: list[int]}，可选) - 设置模型输入的维度，输入维度的顺序与原始模型一致。对于某些模型，模型结构可以进一步优化，但转换后的模型可能会失去动态形状的特征。e.g. {"inTensor1": [1, 32, 32, 32], "inTensor2": [1, 1, 32, 32]}。默认值：""。
        - **input_format** (Format，可选) - 指定导出模型的输入格式。仅对四维输入有效。选项：Format.NHWC | Format.NCHW。默认值：Format.NHWC。
        - **input_data_type** (DataType，可选) - 输入张量的数据类型，默认与模型中定义的类型相同。默认值：DataType.FLOAT32。
        - **output_data_type** (DataType，可选) - 输出张量的数据类型，默认与模型中定义的类型相同。默认值：DataType.FLOAT32。
        - **export_mindir** (ModelType，可选) - 导出模型文件的类型。默认值：ModelType.MINDIR_LITE。
        - **decrypt_key** (str，可选) - 用于解密文件的密钥，以十六进制字符表示。仅当fmk_type为FmkType.MINDIR时有效。默认值：""。
        - **decrypt_mode** (str，可选) - MindIR文件的解密方法。仅在设置decrypt_key时有效。选项："AES-GCM" | "AES-CBC"。默认值："AES-GCM"。
        - **enable_encryption** (bool，可选) - 是否导出加密模型。默认值：False。
        - **encrypt_key** (str，可选) - 用于加密文件的密钥，以十六进制字符表示。仅支持decrypt_mode是"AES-GCM"，密钥长度为16。默认值：""。
        - **infer** (bool，可选) - 转换后是否进行预推理。默认值：False。
        - **train_model** (bool，可选) - 模型是否将在设备上进行训练。默认值：False。
        - **no_fusion** (bool，可选) - 避免融合优化，默认允许融合优化。默认值：False。

    异常：
        - **TypeError** - `fmk_type` 不是FmkType类型。
        - **TypeError** - `model_file` 不是str类型。
        - **TypeError** - `output_file` 不是str类型。
        - **TypeError** - `weight_file` 不是str类型。
        - **TypeError** - `config_file` 不是str类型。
        - **TypeError** - `weight_fp16` 不是bool类型。
        - **TypeError** - `input_shape` 既不是dict类型也不是None。
        - **TypeError** - `input_shape` 是dict类型，但key不是str类型。
        - **TypeError** - `input_shape` 是dict类型，key是str类型，但value不是list类型。
        - **TypeError** - `input_shape` 是dict类型，key是str类型，value是list类型，但value的元素不是int类型。
        - **TypeError** - `input_format` 不是Format类型。
        - **TypeError** - `input_data_type` 不是DataType类型。
        - **TypeError** - `output_data_type` 不是DataType类型。
        - **TypeError** - `export_mindir` 不是ModelType类型。
        - **TypeError** - `decrypt_key` 不是str类型。
        - **TypeError** - `decrypt_mode` 不是str类型。
        - **TypeError** - `enable_encryption` 不是bool类型。
        - **TypeError** - `encrypt_key` 不是str类型。
        - **TypeError** - `infer` 不是bool类型。
        - **TypeError** - `train_model` 不是bool类型。
        - **TypeError** - `no_fusion` 不是bool类型。
        - **ValueError** - 当 `input_format` 是Format类型时， `input_format` 既不是Format.NCHW也不是Format.NHWC。
        - **ValueError** - 当 `decrypt_mode` 是str类型时， `decrypt_mode` 既不是"AES-GCM"也不是"AES-CBC"。
        - **RuntimeError** - `model_file` 文件路径不存在。
        - **RuntimeError** - 当 `model_file` 不是""时， `model_file` 文件路径不存在。
        - **RuntimeError** - 当 `config_file` 不是""时， `config_file` 文件路径不存在。

    .. py:method:: converter()

        执行转换，将第三方模型转换为MindSpore模型。

        异常：
            - **RuntimeError** - 转换模型失败。

    .. py:method:: get_config_info()

        获取转换的配置信息。配套set_config_info方法使用，用于在线推理场景。在get_config_info前，请先用set_config_info方法赋值。

        返回：
            dict{str: dict{str: str}}，在转换中设置的配置信息。

    .. py:method:: set_config_info(section, config_info)

        设置转换时的配置信息。配套get_config_info方法使用，用于在线推理场景。

        参数：
            - **section** (str) - 配置参数的类别。配合config_info一起，设置confile的个别参数。e.g. 对于section是"common_quant_param"，config_info是{"quant_type":"WEIGHT_QUANT"}。默认值：None。
              有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_。
              有关扩展的配置参数，请参见 `extension  <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#%E6%89%A9%E5%B1%95%E9%85%8D%E7%BD%AE>`_。

              - "common_quant_param"：公共量化参数部分。量化的配置参数之一。
              - "mixed_bit_weight_quant_param"：混合位权重量化参数部分。量化的配置参数之一。
              - "full_quant_param"：全量化参数部分。量化的配置参数之一。
              - "data_preprocess_param"：数据预处理参数部分。量化的配置参数之一。
              - "registry"：扩展配置参数部分。扩展的配置参数之一。

            - **config_info** (dict{str: str}，可选) - 配置参数列表。配合section一起，设置confile的个别参数。e.g. 对于section是"common_quant_param"，config_info是{"quant_type":"WEIGHT_QUANT"}。默认值：None。
              有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_。
              有关扩展的配置参数，请参见 `extension  <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#%E6%89%A9%E5%B1%95%E9%85%8D%E7%BD%AE>`_。

        异常：
            - **TypeError** - `section` 不是str类型。
            - **TypeError** - `config_info` 不是dict类型。
            - **TypeError** - `config_info` 是dict类型，但key不是str类型。
            - **TypeError** - `config_info` 是dict类型，key是str类型，但value不是str类型。
