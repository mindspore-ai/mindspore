mindspore.ops.tensordump
========================

.. py:function:: mindspore.ops.tensordump(file_name, tensor, mode='out')

    将tensor保存为npy格式的文件。

    .. warning::
        - 参数 `mode` 将不再支持参数值为'all'。

    在并行的场景下，该算子会保存不同计算卡上的数据分片。

    在昇腾平台的静态图并行模式下，代码 算子A --> 算子B 可能会被编译为 算子A --> 重排算子 --> 算子B。

    注：重排算子指在静态图并行场景中，由于设备间通信，算子切分策略而引入的算子。

    对于 算子A --> 算子B 的情况，算子A计算结果的输出等于算子B的输入。

    然而对于算子A --> 重排算子 --> 算子B这样的情况，由于重排算子的插入，导致算子A的输出不再等价于算子B的输入。

    假设一种情况是算子A的计算结果既作为算子B的输入，也作为tensordump算子的输入。则在该情况下，
    通过设置 `mode` 参数可以实现不同的保存数据的需求：

    - 如果 `mode` 参数设置为'out'，保存的数据仅包含算子A的输出分片。
    - 如果 `mode` 参数设置为'in'，保存的数据将仅包含算子B的输入分片。

    当 `mode` 参数被配置为'in'时，生成的输入分片所对应的npy文件命名格式为：fileName_dumpMode_dtype_id.npy。

    当 `mode` 参数被配置为'out'时，生成的输出分片所对应的npy文件命名格式为：fileName_dtype_id.npy。

    - fileName：参数 `file_name` 的值（若该参数传入时是一个使用者指定的路径，则fileName值为路径的最后一级）。
    - dumpMode：参数 `mode` 的值。
    - dtype：原始的数据类型。bfloat16类型数据保存在.npy文件中会被转换成float32类型。
    - id：一个自增的ID。

    .. note::
        - 在Ascend平台上的graph mode下，可以通过设置环境变量 `MS_DUMP_SLICE_SIZE` 和 `MS_DUMP_WAIT_TIME` 解决在输出大tensor或输出tensor比较密集场景下算子执行失败的问题。
        - 当前该算子不支持在控制流中使用。
        - 如果当前的并行模式为STAND_ALONE，参数 `mode` 只能设置为'out'。
        - 此函数用于调试。

    参数：
        - **file_name** (str) - npy文件的保存路径。
        - **tensor** (Tensor) - 输入的张量。
        - **mode** (str，可选) - 控制tensordump行为模式的参数，可选的值为 ['out', 'in'] 中的一个，默认 ``out``。

    样例：

    .. note::
        使用msrun命令运行下面的例子：msrun \-\-worker_num=2 \-\-local_worker_num=2 \-\-master_port=11450
        \-\-log_dir=msrun_log \-\-join=True \-\-cluster_time_out=300 tensordump_example.py
