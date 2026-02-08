mindspore.parallel.distributed.DistributedDataParallel
============================================================================

.. py:class:: mindspore.parallel.distributed.DistributedDataParallel(module, init_sync=True, process_group=None, bucket_cap_mb=None, find_unused_parameters=False, average_in_collective=False, static_graph=False, reducer_mode="CppReducer")

    分布式数据并行封装类。该类为梯度分配连续显存，各参数的梯度将被分入多个桶，该桶是在数据并行域执行 all-reduce 通信以实现通信掩盖的基本单元。

    .. warning::
        - 该方法当前仅支持在PyNative模式下使用。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **module** (nn.Cell) - 需要进行分布式梯度归约的网络。
        - **init_sync** (bool，可选) - 初始化时，是否进行rank0网络参数广播同步。默认值： ``True``。
        - **process_group** (str，可选) - 梯度归约通信组。默认行为是全局同步。默认值： ``None`` 。
        - **bucket_cap_mb** (int，可选) - 分桶梯度归约的桶大小，单位为MB。不填写时默认采用25MB。默认值： ``None`` 。
        - **find_unused_parameters** (bool，可选) - 是否搜索未使用参数。默认值： ``False`` 。
        - **average_in_collective** (bool，可选) - 是否在通信后求平均，为 ``True`` 时先做AllReduce SUM后scale dp size，否则先做scaling后归约。默认值： ``False`` 。
        - **static_graph** (bool，可选) - 指明是否是静态网络。当是静态网络时，将忽略参数 `find_unused_parameters`，并在第一个step搜索未使用参数，在第二个step前按执行顺序进行桶重建，以实现更好的性能收益。默认值： ``False`` 。
        - **reducer_mode** (str，可选) - 后端梯度归约模式，``"CppReducer"`` 表示采用CPP后端，``"PythonReducer"`` 表示采用Python后端。默认值： ``"CppReducer"`` 。

    返回：
        被 `DistributedDataParallel` 类封装的 `nn.Cell` 网络，网络将自动完成反向梯度归约。

    样例：

    .. note::
        - 使能重计算、梯度冻结时，必须在最外层使用 `DistributedDataParallel` 类进行封装。
        - 在运行以下示例之前，您需要配置通信环境变量。针对Ascend设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/zh-CN/master/parallel/msrun_launcher.html>`_ 。


    .. py:method:: no_sync()

        上下文管理函数，开启时不执行 AllReduce 梯度归约。


    .. py:method:: zero_grad()

        DDP 默认自动累加梯度，手动调用 `zero_grad()` 完成梯度清零。