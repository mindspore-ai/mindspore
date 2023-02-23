mindspore_lite.Converter
========================

.. py:class:: mindspore_lite.Converter()

    构造 `Converter` 的类。使用场景是：1. 将第三方模型转换生成MindSpore模型或MindSpore Lite模型；2. 将MindSpore模型转换生成MindSpore Lite模型。


    .. note::
        请先构造Converter类，再通过执行Converter.converter()方法生成模型。

        加解密功能仅在编译时设置为 `MSLITE_ENABLE_MODEL_ENCRYPTION=on` 时生效，并且仅支持Linux x86平台。其中密钥为十六进制表示的字符串，如encrypt_key设置为"30313233343536373839414243444546"，对应的十六进制表示为 `(b)0123456789ABCDEF` ，Linux平台用户可以使用 `xxd` 工具对字节表示的密钥进行十六进制表达转换。需要注意的是，加解密算法在1.7版本进行了更新，导致新版的python接口不支持对1.6及其之前版本的MindSpore Lite加密导出的模型进行转换。

    .. py:method:: weight_fp16
        :property:

        获取模型是否保存为Floa16数据类型。

    .. py:method:: input_shape
        :property:

        获取模型输入的维度。

    .. py:method:: input_format
        :property:

        获取模型的输入format。

    .. py:method:: input_data_type
        :property:

        获取量化模型输入Tensor的数据类型。

    .. py:method:: output_data_type
        :property:

        获取量化模型输出tensor的data type。

    .. py:method:: save_type
        :property:

        获取导出模型文件的类型。

    .. py:method:: decrypt_key
        :property:

        获取用于加载密文MindIR时的密钥。

    .. py:method:: decrypt_mode
        :property:

        获取加载密文MindIR的模式。

    .. py:method:: enable_encryption
        :property:

        导出模型时是否加密。

    .. py:method:: encrypt_key
        :property:

        获取用于加密文件的密钥，以十六进制字符表示。

    .. py:method:: infer
        :property:

        Converter后是否进行预推理。

    .. py:method:: train_model
        :property:

        模型是否将在设备上进行训练。

    .. py:method:: optimize
        :property:

        是否避免融合优化。

        .. note::
            - optimize是用来设定在离线转换的过程中需要完成哪些特定的优化。如果该参数设置为"none"，那么在模型的离线转换阶段将不进行相关的图优化操作，相关的图优化操作将会在执行推理阶段完成。该参数的优点在于转换出来的模型由于没有经过特定的优化，可以直接部署到CPU/GPU/Ascend任意硬件后端；而带来的缺点是推理执行时模型的初始化时间增长。如果设置成"general"，表示离线转换过程会完成通用优化，包括常量折叠，算子融合等（转换出的模型只支持CPU/GPU后端，不支持Ascend后端）。如果设置成"ascend_oriented"，表示转换过程中只完成针对Ascend后端的优化（转换出来的模型只支持Ascend后端）。

            - 针对MindSpore模型，由于已经是mindir模型，建议两种做法：

              - 不需要经过离线转换，直接进行推理执行。
              - 使用离线转换，CPU/GPU后端设置optimize为"general"，NPU后端设置optimize为"ascend_oriented"，在离线阶段完成相关优化，减少推理执行的初始化时间。

    .. py:method:: device
        :property:

        设置转换模型时的目标设备。

    .. py:method:: converter(fmk_type, model_file, output_file="", weight_file="", config_file="")

        执行转换，将第三方模型转换为MindSpore模型。

        参数：
            - **fmk_type** (FmkType) - 输入模型框架类型。选项：FmkType.TF | FmkType.CAFFE | FmkType.ONNX | FmkType.MINDIR | FmkType.TFLITE | FmkType.PYTORCH。有关详细信息，请参见 `FmkType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.FmkType.html>`_ 。
            - **model_file** (str) - 转换时的输入模型文件路径。例如："/home/user/model.prototxt"。选项：TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | MINDIR: "model.mindir" | TFLITE: "model.tflite" | PYTORCH: "model.pt or model.pth"。
            - **output_file** (str) - 转换时的输出模型文件路径。可自动生成.ms后缀。如果将 `save_type` 设置为ModelType.MINDIR，那么将生成MindSpore模型，该模型使用.mindir作为后缀。如果将 `save_type` 设置为ModelType.MINDIR_LITE，那么将生成MindSpore Lite模型，该模型使用.ms作为后缀。例如：输入模型为"/home/user/model.prototxt"，它将生成名为model.prototxt.ms的模型在/home/user/路径下。
            - **weight_file** (str，可选) - 输入模型权重文件。仅当输入模型框架类型为FmkType.CAFFE时必选，Caffe模型一般分为两个文件： `model.prototxt` 是模型结构，对应 `model_file` 参数； `model.caffemodel` 是模型权值文件，对应 `weight_file` 参数。例如："/home/user/model.caffemodel"。默认值：""。
            - **config_file** (str，可选) - Converter的配置文件，可配置训练后量化或离线拆分算子并行或禁用算子融合功能并将插件设置为so路径等功能。 `config_file` 配置文件采用 `key = value` 的方式定义相关参数，有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。有关扩展的配置参数，请参见 `extension <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。例如："/home/user/model.cfg"。默认值：""。

        异常：
            - **TypeError** - `fmk_type` 不是FmkType类型。
            - **TypeError** - `model_file` 不是str类型。
            - **TypeError** - `output_file` 不是str类型。
            - **TypeError** - `weight_file` 不是str类型。
            - **TypeError** - `config_file` 不是str类型。
            - **RuntimeError** - `model_file` 文件路径不存在。
            - **RuntimeError** - 当 `model_file` 不是""时， `model_file` 文件路径不存在。
            - **RuntimeError** - 当 `config_file` 不是""时， `config_file` 文件路径不存在。
            - **RuntimeError** - 转换模型失败。

    .. py:method:: get_config_info()

        获取Converter时的配置信息。配套 `set_config_info` 方法使用，用于在线推理场景。在 `get_config_info` 前，请先用 `set_config_info` 方法赋值。

        返回：
            dict{str: dict{str: str}}，在Converter中设置的配置信息。

    .. py:method:: set_config_info(section="", config_info=None)

        设置Converter时的配置信息。配套 `get_config_info` 方法使用，用于在线推理场景。

        参数：
            - **section** (str，可选) - 配置参数的类别。配合 `config_info` 一起，设置confile的个别参数。例如：对于 `section` 是"common_quant_param"， `config_info` 是{"quant_type":"WEIGHT_QUANT"}。默认值：""。

              有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。

              有关扩展的配置参数，请参见 `extension <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。

              - "common_quant_param"：公共量化参数部分。
              - "mixed_bit_weight_quant_param"：混合位权重量化参数部分。
              - "full_quant_param"：全量化参数部分。
              - "data_preprocess_param"：数据预处理量化参数部分。
              - "registry"：扩展配置参数部分。

            - **config_info** (dict{str: str}，可选) - 配置参数列表。配合 `section` 一起，设置confile的个别参数。例如：对于 `section` 是"common_quant_param"， `config_info` 是{"quant_type":"WEIGHT_QUANT"}。默认值：None。

              有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。

              有关扩展的配置参数，请参见 `extension <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。

        异常：
            - **TypeError** - `section` 不是str类型。
            - **TypeError** - `config_info` 不是dict类型。
            - **TypeError** - `config_info` 是dict类型，但key不是str类型。
            - **TypeError** - `config_info` 是dict类型，key是str类型，但value不是str类型。
