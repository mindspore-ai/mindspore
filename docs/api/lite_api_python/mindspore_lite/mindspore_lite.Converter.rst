mindspore_lite.Converter
========================

.. py:class:: mindspore_lite.Converter()

    构造 `Converter` 的类。使用场景是：1. 将第三方模型转换生成MindSpore模型或MindSpore Lite模型；2. 将MindSpore模型转换生成MindSpore Lite模型。


    .. note::
        请先构造Converter类，再通过执行Converter.converter()方法生成模型。

        加解密功能仅在编译时设置为 `MSLITE_ENABLE_MODEL_ENCRYPTION=on` 时生效，并且仅支持Linux x86平台。其中密钥为十六进制表示的字符串，如encrypt_key设置为"30313233343536373839414243444546"，对应的十六进制表示为 `(b)0123456789ABCDEF` ，Linux平台用户可以使用 `xxd` 工具对字节表示的密钥进行十六进制表达转换。需要注意的是，加解密算法在1.7版本进行了更新，导致新版的Python接口不支持对1.6及其之前版本的MindSpore Lite加密导出的模型进行转换。

    .. py:method:: converter(fmk_type, model_file, output_file, weight_file="", config_file="")

        执行转换，将第三方模型转换为MindSpore模型。

        参数：
            - **fmk_type** (FmkType) - 输入模型框架类型。选项：FmkType.TF | FmkType.CAFFE | FmkType.ONNX | FmkType.MINDIR | FmkType.TFLITE | FmkType.PYTORCH。有关详细信息，请参见 `框架类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.FmkType.html>`_ 。
            - **model_file** (str) - 转换时的输入模型文件路径。例如："/home/user/model.prototxt"。选项：TF: "model.pb" | CAFFE: "model.prototxt" | ONNX: "model.onnx" | MINDIR: "model.mindir" | TFLITE: "model.tflite" | PYTORCH: "model.pt or model.pth"。
            - **output_file** (str) - 转换时的输出模型文件路径。可自动生成.ms后缀。如果将 `save_type` 设置为ModelType.MINDIR，那么将生成MindSpore模型，该模型使用.mindir作为后缀。如果将 `save_type` 设置为ModelType.MINDIR_LITE，那么将生成MindSpore Lite模型，该模型使用.ms作为后缀。例如：输入模型为"/home/user/model.prototxt"，它将生成名为model.prototxt.ms的模型在/home/user/路径下。
            - **weight_file** (str，可选) - 输入模型权重文件。仅当输入模型框架类型为FmkType.CAFFE时必选，Caffe模型一般分为两个文件： `model.prototxt` 是模型结构，对应 `model_file` 参数； `model.caffemodel` 是模型权值文件，对应 `weight_file` 参数。例如："/home/user/model.caffemodel"。默认值：""，表示无模型权重文件。
            - **config_file** (str，可选) - Converter的配置文件，可配置训练后量化或离线拆分算子并行或禁用算子融合功能并将插件设置为so路径等功能。 `config_file` 配置文件采用 `key = value` 的方式定义相关参数，有关训练后量化的配置参数，请参见 `训练后量化 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。有关扩展的配置参数，请参见 `扩展配置 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。例如："/home/user/model.cfg"。默认值：""，表示不设置Converter的配置文件。

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

    .. py:method:: decrypt_key
        :property:

        获取用于加载密文MindIR时的密钥。

        返回：
            str，用于加载密文MindIR时的密钥，以十六进制字符表示。仅当fmk_type为FmkType.MINDIR时有效。

    .. py:method:: decrypt_mode
        :property:

        获取加载密文MindIR的模式。

        返回：
            str，加载密文MindIR的模式。只在设置了 `decryptKey` 时有效。选项："AES-GCM" | "AES-CBC"。

    .. py:method:: device
        :property:

        获取转换模型时的目标设备。

        返回：
            str，转换模型时的目标设备。仅对Ascend设备有效。使用场景是在Ascend设备上，如果你需要转换生成的模型调用Ascend后端执行推理，则设置该参数，若未设置，默认模型调用CPU后端推理。支持以下目标设备："Ascend"。

    .. py:method:: enable_encryption
        :property:

        获取导出模型时是否加密的状态。

        返回：
            bool，导出模型时是否加密。导出加密可保护模型完整性，但会增加运行时初始化时间。

    .. py:method:: encrypt_key
        :property:

        获取用于加密文件的密钥。

        返回：
            str，用于加密文件的密钥，以十六进制字符表示。仅支持当 `decrypt_mode` 是"AES-GCM"，密钥长度为16。

    .. py:method:: get_config_info()

        获取Converter时的配置信息。配套 `set_config_info` 方法使用，用于在线推理场景。在 `get_config_info` 前，请先用 `set_config_info` 方法赋值。

        返回：
            dict{str: dict{str: str}}，在Converter中设置的配置信息。

    .. py:method:: infer
        :property:

        获取Converter后是否进行预推理的状态。

        返回：
            bool，Converter后是否进行预推理。

    .. py:method:: input_data_type
        :property:

        获取量化模型输入Tensor的数据类型。

        返回：
            DataType，量化模型输入Tensor的数据类型。仅当模型输入Tensor的量化参数（ `scale` 和 `zero point` ）都具备时有效。默认与原始模型输入Tensor的data type保持一致。支持以下4种数据类型：DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 | DataType.UNKNOWN。默认值：DataType.FLOAT32。有关详细信息，请参见 `数据类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.DataType.html>`_ 。

            - **DataType.FLOAT32** - 32位浮点数。
            - **DataType.INT8**    - 8位整型数。
            - **DataType.UINT8**   - 无符号8位整型数。
            - **DataType.UNKNOWN** - 设置与模型输入Tensor相同的DataType。

    .. py:method:: input_format
        :property:

        获取模型的输入format。

        返回：
            Format，模型的输入format。仅对四维输入有效。支持以下2种输入格式：Format.NCHW | Format.NHWC。默认值：Format.NHWC。有关详细信息，请参见 `数据格式 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.Format.html>`_ 。

            - **Format.NCHW** - 按批次N、通道C、高度H和宽度W的顺序存储Tensor数据。
            - **Format.NHWC** - 按批次N、高度H、宽度W和通道C的顺序存储Tensor数据。

    .. py:method:: input_shape
        :property:

        获取模型输入的维度。

        返回：
            dict{str, list[int]}，模型输入的维度。输入维度的顺序与原始模型一致。在以下场景下，用户可能需要设置该参数。例如：{"inTensor1": [1, 32, 32, 32], "inTensor2": [1, 1, 32, 32]}。默认值：None，等同于设置为{}。

            - **用法1** - 待转换模型的输入是动态shape，准备采用固定shape推理，则设置该参数为固定shape。设置之后，在对Converter后的模型进行推理时，默认输入的shape与该参数设置一样，无需再进行resize操作。
            - **用法2** - 无论待转换模型的原始输入是否为动态shape，准备采用固定shape推理，并希望模型的性能尽可能优化，则设置该参数为固定shape。设置之后，将对模型结构进一步优化，但转换后的模型可能会失去动态shape的特征（部分跟shape强相关的算子会被融合）。
            - **用法3** - 使用Converter功能来生成用于Micro推理执行代码时，推荐配置该参数，以减少部署过程中出错的概率。当模型含有Shape算子或者待转换模型输入为动态shape时，则必须配置该参数，设置固定shape，以支持相关shape优化和代码生成。

    .. py:method:: optimize
        :property:

        获取是否融合优化的状态。

        optimize是用来设定在离线转换的过程中需要完成哪些特定的优化。如果该参数设置为"none"，那么在模型的离线转换阶段将不进行相关的图优化操作，相关的图优化操作将会在执行推理阶段完成。该参数的优点在于转换出来的模型由于没有经过特定的优化，可以直接部署到CPU/GPU/Ascend任意硬件后端；而带来的缺点是推理执行时模型的初始化时间增长。如果设置成"general"，表示离线转换过程会完成通用优化，包括常量折叠，算子融合等（转换出的模型只支持CPU/GPU后端，不支持Ascend后端）。如果设置成"ascend_oriented"，表示转换过程中只完成针对Ascend后端的优化（转换出来的模型只支持Ascend后端）。

        .. note::
            针对MindSpore模型，由于已经是mindir模型，建议两种做法：

              - 不需要经过离线转换，直接进行推理执行。
              - 使用离线转换，CPU/GPU后端设置optimize为"general"，NPU后端设置optimize为"ascend_oriented"，在离线阶段完成相关优化，减少推理执行的初始化时间。

        返回：
            str，是否融合优化。选项："none" | "general" | "ascend_oriented"。"none" 表示不允许融合优化。 "general" 和 "ascend_oriented" 表示允许融合优化。

    .. py:method:: output_data_type
        :property:

        获取量化模型输出Tensor的data type。

        返回：
            DataType，量化模型输出Tensor的data type。仅当模型输出Tensor的量化参数（scale和zero point）都具备时有效。默认与原始模型输出Tensor的data type保持一致。支持以下4种数据类型：DataType.FLOAT32 | DataType.INT8 | DataType.UINT8 | DataType.UNKNOWN。有关详细信息，请参见 `数据类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.DataType.html>`_ 。

            - **DataType.FLOAT32** - 32位浮点数。
            - **DataType.INT8**    - 8位整型数。
            - **DataType.UINT8**   - 无符号8位整型数。
            - **DataType.UNKNOWN** - 设置与模型输出Tensor相同的DataType。

    .. py:method:: save_type
        :property:

        获取导出模型文件的类型。

        返回：
            ModelType，导出模型文件的类型。选项：ModelType.MINDIR | ModelType.MINDIR_LITE。有关详细信息，请参见 `模型类型 <https://mindspore.cn/lite/api/zh-CN/master/mindspore_lite/mindspore_lite.ModelType.html>`_ 。

    .. py:method:: set_config_info(section="", config_info=None)

        设置Converter时的配置信息。配套 `get_config_info` 方法使用，用于在线推理场景。

        参数：
            - **section** (str，可选) - 配置参数的类别。配合 `config_info` 一起，设置confile的个别参数。例如：对于 `section` 是"common_quant_param"， `config_info` 是{"quant_type":"WEIGHT_QUANT"}。默认值：""。

              有关训练后量化的配置参数，请参见 `训练后量化 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。

              有关扩展的配置参数，请参见 `扩展配置 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。

              - "common_quant_param"：公共量化参数部分。
              - "mixed_bit_weight_quant_param"：混合位权重量化参数部分。
              - "full_quant_param"：全量化参数部分。
              - "data_preprocess_param"：数据预处理量化参数部分。
              - "registry"：扩展配置参数部分。

            - **config_info** (dict{str: str}，可选) - 配置参数列表。配合 `section` 一起，设置confile的个别参数。例如：对于 `section` 是"common_quant_param"， `config_info` 是{"quant_type":"WEIGHT_QUANT"}。默认值：None。

              有关训练后量化的配置参数，请参见 `训练后量化 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/post_training_quantization.html>`_ 。

              有关扩展的配置参数，请参见 `扩展配置 <https://www.mindspore.cn/lite/docs/zh-CN/master/use/nnie.html#扩展配置>`_ 。

        异常：
            - **TypeError** - `section` 不是str类型。
            - **TypeError** - `config_info` 不是dict类型。
            - **TypeError** - `config_info` 是dict类型，但key不是str类型。
            - **TypeError** - `config_info` 是dict类型，key是str类型，但value不是str类型。

    .. py:method:: train_model
        :property:

        获取模型是否将在设备上进行训练的状态。

        .. note::
            此属性不支持在MindSpore Lite云侧推理包上使用。

        返回：
            bool，模型是否将在设备上进行训练。

    .. py:method:: weight_fp16
        :property:

        获取模型是否保存为Float16数据类型的状态。

        返回：
            bool，模型是否保存为Float16数据类型。若True，则在转换时，会将模型中Float32的常量Tensor保存成Float16数据类型，压缩生成的模型尺寸。之后根据 `Context.CPU` 的 `precision_mode` 参数决定输入的数据类型执行推理。 `weight_fp16` 的优先级很低，比如如果开启了量化，那么对于已经量化的权重， `weight_fp16` 不会再次生效。 `weight_fp16` 仅对Float32数据类型中的常量Tensor有效。
