mindspore.parallel.unified_safetensors
======================================

.. py:function:: mindspore.parallel.unified_safetensors(src_dir, src_strategy_file, dst_dir, merge_with_redundancy=True, file_suffix=None, max_process_num=64, choice_func=None, split_dst_file=())

    将多个safetensors文件合并为一系列统一的safetensors文件。

    .. note::
        权重合并时，校验入参 `merge_with_redundancy` 与所合并的safetensors文件中的去冗余标志位是否不同。如果相同，按照文件的去冗余标志位进行合并。

    参数：
        - **src_dir** (str) - 源权重保存目录。
        - **src_strategy_file** (str) - 源权重切分策略文件，文件扩展名是 `.ckpt` 。
        - **dst_dir** (str) - 目标保存目录。
        - **merge_with_redundancy** (bool, 可选) - 合并源权重文件是否是去冗余保存的safetensors文件。默认值是：``True``，合并的源权重文件是完整的。
        - **file_suffix** (str，可选) - 指定合并safetensors的文件名后缀。默认值：``None`` ，合并源权重目录下所有的safetensors文件。
        - **max_process_num** (int，可选) - 最大进程数。默认值： ``64``。
        - **choice_func** (callable，可选) - 可调用的函数，用于筛选参数或者修改参数名，函数的返回值必须为str或者bool类型。默认值：``None``。
        - **split_dst_file** (tuple，可选) - 手动将任务切分为多个子任务执行，以元组形式表示，元组中包含两个元素：第一个元素表示当前子任务编号，第二个元素表示任务的总数量。该参数支持在单台机器上多次切分所执行的任务，也支持在多台机器上分别执行不同的子任务。默认值：``()``。

    异常：
        - **ValueError** - 如果某个rank的safetensors文件丢失。
