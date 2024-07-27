mindspore.ckpt_to_safetensors
======================================

.. py:function:: mindspore.ckpt_to_safetensors(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1)

    将 MindSpore 的 checkpoint 文件转换为 safetensors 格式并保存到 `save_path`。

    .. note::
        多进程设置数量与主机规模有关，不推荐设置太大，否则容易导致卡死。

    参数：
        - **file_path** (str) - 包含 checkpoint 文件的目录路径或单个 checkpoint 文件 (.ckpt) 的路径。
        - **save_path** (str, 可选) - 保存 safetensors 文件的目录路径。默认值：``None``。
        - **name_map** (dict, 可选) - 映射原始参数名到新参数名的字典。默认值：``None``。
        - **file_name_regex** (str, 可选) - 用于匹配需要转换的文件的正则表达式。默认值：``None``。
        - **processes_num** (int, 可选) - 控制并行处理的进程数量。默认值： ``1``。

    异常：
        - **ValueError** - 如果输入路径无效，或 save_path 不是目录，或 file_path 不以 '.ckpt' 结尾。
