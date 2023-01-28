mindspore_lite.Converter
========================

.. py:class:: mindspore_lite.Converter(fmk_type, model_file, output_file, weight_file="", config_file="", weight_fp16=False, input_shape=None, input_format=Format.NHWC, input_data_type=DataType.FLOAT32, output_data_type=DataType.FLOAT32, export_mindir=ModelType.MINDIR_LITE, decrypt_key="", decrypt_mode="AES-GCM", enable_encryption=False, encrypt_key="", infer=False, train_model=False, no_fusion=None, device="")

    构造 `Converter` 的类。使用场景是：1. 将第三方模型转换生成MindSpore模型或MindSpore Lite模型；2. 将MindSpore模型转换生成MindSpore Lite模型。


    .. note::
        请先构造Converter类，再通过执行Converter.converter()方法生成模型。

        加解密功能仅在编译时设置为 `MSLITE_ENABLE_MODEL_ENCRYPTION=on` 时生效，并且仅支持Linux x86平台。其中密钥为十六进制表示的字符串，如密钥定义为 `(b)0123456789ABCDEF` 对应的十六进制表示为 `30313233343536373839414243444546` ，Linux平台用户可以使用 `xxd` 工具对字节表示的密钥进行十六进制表达转换。需要注意的是，加解密算法在1.7版本进行了更新，导致新版的python接口不支持对1.6及其之前版本的MindSpore Lite加密导出的模型进行转换。

    参数：
        - **fmk_type** (FmkType) - 输入模型框架类型。选项：FmkType.TF | FmkType.CAFFE | FmkType.ONNX | FmkType.MINDIR | FmkType.TFLITE | FmkType.PYTORCH。有关详细信息，请参见 `FmkType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.FmkType.html>`_ 。
        - **model_file** (str) - 转换时的输入模型文件路径。例如："/home/user/model.prototxt"。选项：TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | MINDIR: "model.mindir" | TFLITE: "model.tflite" | PYTORCH: "model.pt or model.pth"。
        - **output_file** (str) - 转换时的输出模型文件路径。可自动生成.ms后缀。如果将 `export_mindir` 设置为ModelType.MINDIR，那么将生成MindSpore模型，该模型使用.mindir作为后缀。如果将 `export_mindir` 设置为ModelType.MINDIR_LITE，那么将生成MindSpore Lite模型，该模型使用.ms作为后缀。例如：输入模型为"/home/user/model.prototxt"，它将生成名为model.prototxt.ms的模型在/home/user/路径下。
        - **weight_file** (str，可选) - 输入模型权重文件。仅当输入模型框架类型为FmkType.CAFFE时必选，Caffe模型一般分为两个文件： `model.prototxt` 是模型结构，对应 `model_file` 参数； `model.caffemodel` 是模型权值文件，对应 `weight_file` 参数。例如："/home/user/model.caffemodel"。默认值：""。
        - **config_file** (str，可选) - Converter的配置文件，可配置训练后量化或离线拆分算子并行或禁用算子融合功能并将插件设置为so路径等功能。 `config_file` 配置文件采用 `key = value` 的方式定义相关参数，有关训练后量化的配置参数，请参见 `quantization <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。有关扩展的配置参数，请参见 `extension <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。例如："/home/user/model.cfg"。默认值：""。
        - **weight_fp16** (bool，可选) - 若True，则在转换时，会将模型中Float32的常量Tensor保存成Float16数据类型，压缩生成的模型尺寸。之后根据 `DeviceInfo` 的 `enable_fp16` 参数决定输入的数据类型执行推理。 `weight_fp16` 的优先级很低，比如如果开启了量化，那么对于已经量化的权重， `weight_fp16` 不会再次生效。 `weight_fp16` 仅对Float32数据类型中的常量Tensor有效。默认值：False。
        - **input_shape** (dict{str: list[int]}，可选) - 设置模型输入的维度，输入维度的顺序与原始模型一致。在以下场景下，用户可能需要设置该参数。例如：{"inTensor1": [1, 32, 32, 32], "inTensor2": [1, 1, 32, 32]}。默认值：None，等同于设置为{}。

          - **用法1** - 待转换模型的输入是动态shape，准备采用固定shape推理，则设置该参数为固定shape。设置之后，在对Converter后的模型进行推理时，默认输入的shape与该参数设置一样，无需再进行resize操作。
          - **用法2** - 无论待转换模型的原始输入是否为动态shape，准备采用固定shape推理，并希望模型的性能尽可能优化，则设置该参数为固定shape。设置之后，将对模型结构进一步优化，但转换后的模型可能会失去动态shape的特征（部分跟shape强相关的算子会被融合）。
          - **用法3** - 使用Converter功能来生成用于Micro推理执行代码时，推荐配置该参数，以减少部署过程中出错的概率。当模型含有Shape算子或者待转换模型输入为动态shape时，则必须配置该参数，设置固定shape，以支持相关shape优化和代码生成。

        - **input_format** (Format，可选) - 设置导出模型的输入format。仅对四维输入有效。支持以下2种输入格式：Format.NCHW | Format.NHWC。默认值：Format.NHWC。有关详细信息，请参见 `Format <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Format.html>`_ 。

          - **Format.NCHW** - 按批次N、通道C、高度H和宽度W的顺序存储Tensor数据。
          - **Format.NHWC** - 按批次N、高度H、宽度W和通道C的顺序存储Tensor数据。

        - **input_data_type** (DataType，可选) - 设置量化模型输入Tensor的数据类型。仅当模型输入tensor的量化参数（ `scale` 和 `zero point` ）都具备时有效。默认与原始模型输入tensor的data type保持一致。支持以下4种数据类型：DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 | DataType.UNKNOWN。默认值：DataType.FLOAT32。有关详细信息，请参见 `DataType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.DataType.html>`_ 。

          - **DataType.FLOAT32** - 32位浮点数。
          - **DataType.INT8**    - 8位整型数。
          - **DataType.UINT8**   - 无符号8位整型数。
          - **DataType.UNKNOWN** - 设置与模型输入Tensor相同的DataType。

        - **output_data_type** (DataType，可选) - 设置量化模型输出tensor的data type。仅当模型输出tensor的量化参数（scale和zero point）都具备时有效。默认与原始模型输出tensor的data type保持一致。支持以下4种数据类型：DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 | DataType.UNKNOWN。默认值：DataType.FLOAT32。有关详细信息，请参见 `DataType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.DataType.html>`_ 。

          - **DataType.FLOAT32** - 32位浮点数。
          - **DataType.INT8**    - 8位整型数。
          - **DataType.UINT8**   - 无符号8位整型数。
          - **DataType.UNKNOWN** - 设置与模型输出Tensor相同的DataType。

        - **export_mindir** (ModelType，可选) - 设置导出模型文件的类型。选项：ModelType.MINDIR | ModelType.MINDIR_LITE。默认值：ModelType.MINDIR_LITE。有关详细信息，请参见 `ModelType <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.ModelType.html>`_ 。
        - **decrypt_key** (str，可选) - 设置用于加载密文MindIR时的密钥，以十六进制字符表示。仅当fmk_type为FmkType.MINDIR时有效。默认值：""。
        - **decrypt_mode** (str，可选) - 设置加载密文MindIR的模式，只在设置了 `decryptKey` 时有效。选项："AES-GCM" | "AES-CBC"。默认值："AES-GCM"。
        - **enable_encryption** (bool，可选) - 导出模型时是否加密，导出加密可保护模型完整性，但会增加运行时初始化时间。默认值：False。
        - **encrypt_key** (str，可选) - 设置用于加密文件的密钥，以十六进制字符表示。仅支持当 `decrypt_mode` 是"AES-GCM"，密钥长度为16。默认值：""。
        - **infer** (bool，可选) - Converter后是否进行预推理。默认值：False。
        - **train_model** (bool，可选) - 模型是否将在设备上进行训练。默认值：False。
        - **no_fusion** (bool，可选) - 是否避免融合优化。默认值：False。当 `export_mindir` 为ModelType.mindir时，None为True，意味着避免融合优化。当 `export_mindir` 不是ModelType.mindir时，None为False，意味着允许融合优化。
        - **device** (str，可选) - 设置转换模型时的目标设备，仅对Ascend设备有效。使用场景是在Ascend设备上，如果你需要转换生成的模型调用Ascend后端执行推理，则设置该参数，若未设置，默认模型调用CPU后端推理。支持以下目标设备："Ascend"。默认值：""。

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
        - **TypeError** - `device` 不是str类型。
        - **ValueError** - 当 `input_format` 是Format类型时， `input_format` 既不是Format.NCHW也不是Format.NHWC。
        - **ValueError** - 当 `decrypt_mode` 是str类型时， `decrypt_mode` 既不是"AES-GCM"也不是"AES-CBC"。
        - **ValueError** - 当 `device` 是str类型时， `device` 不是"Ascend"。
        - **RuntimeError** - `model_file` 文件路径不存在。
        - **RuntimeError** - 当 `model_file` 不是""时， `model_file` 文件路径不存在。
        - **RuntimeError** - 当 `config_file` 不是""时， `config_file` 文件路径不存在。

    .. py:method:: converter()

        执行转换，将第三方模型转换为MindSpore模型。

        异常：
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
