mindspore.ops.Primitive
=======================

.. py:class:: mindspore.ops.Primitive(name)

    Primitive是Python中算子原语的基类。

    参数：
        - **name** (str) - 当前Primitive的名称。

    .. py:method:: add_prim_attr(name, value)

        添加Primitive的属性。

        参数：
            - **name** (str) - 属性名称。
            - **value** (Any) - 属性值。

    .. py:method:: check_elim(*args)

        检查是否可以消除此Primitive。有需要的子类可以重写该方法。

        参数：
            - **args** (Primitive args) - 与当前Primitive的参数相同。

        返回：
            由两个元素组成的元组。第一个元素是指是否能在编译阶段计算Primitive，第二个元素是计算结果。

    .. py:method:: del_prim_attr(name)

        删除Primitive的属性。

        参数：
            - **name** (str) - 属性名称。

    .. py:method:: init_prim_io_names(inputs, outputs)

        初始化Tensor或属性的输入输出的名称。

        参数：
            - **inputs** (list[str]) - 输入名称的列表。
            - **outputs** (list[str]) - 输出名称的列表。

    .. py:method:: place(role, rank_id)
        
        设置Primitive算子标签。让此算子在对应进程上执行。
        每个进程的标签都由进程角色 `role` 和 `rank_id` 组成，因此对算子设置标签，代表着算子在哪个进程执行，从而执行分布式训练等任务。

        .. note::
            - 此接口只在成功调用 `mindspore.communication.init()` 完成动态组网后才能生效。

        参数：
            - **role** (str) - 算子执行所在进程的角色。只支持'MS_WORKER'。
            - **rank_id** (int) - 算子执行所在进程的id。在相同角色进程间， `rank_id` 是唯一的。

    .. py:method:: recompute(mode=True)

        设置Primitive的重计算属性。

        如果有一个被设置了重计算属性的Primitive，并且其结果在计算导数的时候被使用，那么不会保存该Primitive在前向网络中的中间计算结果，而是在自动微分的时候重新进行计算。

        .. note::
            - 如果计算涉及随机化或全局变量，则暂无法保证等效性。
            - 在PyNative模式下不支持。

        参数：
            - **mode** (bool) - Primitive是否设置了重计算。默认值：True。

    .. py:method:: set_device(device_target)

        设置Primitive执行后端。

        参数：
            - **device_target** (str) - 后端名称，支持CPU、GPU、Ascend。

    .. py:method:: set_prim_instance_name(instance_name)

        设置Primitive算子的实例的名称。

        .. note::
            当用户定义Primitive算子时，默认调用它。

        参数：
            - **instance_name** (str) - 用户设置的Primitive算子的实例的名称。

    .. py:method:: set_stage(stage)

        将stage的ID添加到Primitive属性中。

        .. note::
            仅在半自动并行模式下有效。在其他并行模式下，请将其设置为0。

        参数：
            - **stage** (int) - 当前stage的ID。

    .. py:method:: shard(in_strategy=None, out_strategy=None)

        将切分策略添加到Primitive属性中。

        .. note::
            仅在半自动并行或自动并行模式下有效。在其他并行模式中，将忽略此处设置的策略。

        参数：
            - **in_strategy** (tuple) - 描述算子输入的切分策略。默认值：None。
            - **out_strategy** (tuple) - 描述算子输出的切分策略，仅针对某些算子，如MatMul。默认值：None。

    .. py:method:: update_parameter()
        :property:

        判断此Primitive是否会更新参数的值。
