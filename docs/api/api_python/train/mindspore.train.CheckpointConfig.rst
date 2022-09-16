mindspore.train.CheckpointConfig
================================

.. py:class:: mindspore.train.CheckpointConfig(save_checkpoint_steps=1, save_checkpoint_seconds=0, keep_checkpoint_max=5, keep_checkpoint_per_n_minutes=0, integrated_save=True, async_save=False, saved_network=None, append_info=None, enc_key=None, enc_mode='AES-GCM', exception_save=False)

    保存checkpoint时的配置策略。

    .. note::
        在训练过程中，如果数据集是通过数据通道传输的，建议将 `save_checkpoint_steps` 设为循环下沉step数量的整数倍数。否则，保存checkpoint的时机可能会有偏差。建议同时只设置一种触发保存checkpoint策略和一种保留checkpoint文件总数策略。如果同时设置了 `save_checkpoint_steps` 和 `save_checkpoint_seconds` ，则 `save_checkpoint_seconds` 无效。如果同时设置了 `keep_checkpoint_max` 和 `keep_checkpoint_per_n_minutes` ，则 `keep_checkpoint_per_n_minutes` 无效。

    参数：
        - **save_checkpoint_steps** (int) - 每隔多少个step保存一次checkpoint。默认值：1。
        - **save_checkpoint_seconds** (int) - 每隔多少秒保存一次checkpoint。不能同时与 `save_checkpoint_steps` 一起使用。默认值：0。
        - **keep_checkpoint_max** (int) - 最多保存多少个checkpoint文件。默认值：5。
        - **keep_checkpoint_per_n_minutes** (int) - 每隔多少分钟保存一个checkpoint文件。不能同时与 `keep_checkpoint_max` 一起使用。默认值：0。
        - **integrated_save** (bool) - 在自动并行场景下，是否合并保存拆分后的Tensor。合并保存功能仅支持在自动并行场景中使用，在手动并行场景中不支持。默认值：True。
        - **async_save** (bool) - 是否异步执行保存checkpoint文件。默认值：False。
        - **saved_network** (Cell) - 保存在checkpoint文件中的网络。如果 `saved_network` 没有被训练，则保存 `saved_network` 的初始值。默认值：None。
        - **append_info** (list) - 保存在checkpoint文件中的信息。支持"epoch_num"、"step_num"和dict类型。dict的key必须是str，dict的value必须是int、float、bool、string、Parameter或Tensor中的一个。默认值：None。
        - **enc_key** (Union[None, bytes]) - 用于加密的字节类型key。如果值为None，则不需要加密。默认值：None。
        - **enc_mode** (str) - 仅当 `enc_key` 不设为None时，该参数有效。指定了加密模式，目前支持AES-GCM和AES-CBC。默认值：AES-GCM。
        - **exception_save** (bool) - 当有异常发生时，是否保存当前checkpoint文件。默认值：False。

    异常：
        - **ValueError** - 输入参数的类型不正确。

    .. py:method:: append_dict
        :property:

        获取需要额外保存到checkpoint中的字典的值。

        返回：
            Dict: 字典中的值。

    .. py:method:: async_save
        :property:

        获取是否异步保存checkpoint。

        返回：
            Bool: 是否异步保存checkpoint。

    .. py:method:: enc_key
        :property:

        获取加密的key值。

        返回：
            (None, bytes): 加密的key值。

    .. py:method:: enc_mode
        :property:

        获取加密模式。

        返回：
            str: 加密模式。

    .. py:method:: get_checkpoint_policy()

        获取checkpoint的保存策略。

        返回：
            Dict: checkpoint的保存策略。

    .. py:method:: integrated_save
        :property:

        获取是否合并保存拆分后的Tensor。

        返回：
            Bool: 获取是否合并保存拆分后的Tensor。

    .. py:method:: keep_checkpoint_max
        :property:

        获取最多保存checkpoint文件的数量。

        返回：
            Int: 最多保存checkpoint文件的数量。

    .. py:method:: keep_checkpoint_per_n_minutes
        :property:

        获取每隔多少分钟保存一个checkpoint文件。

        返回：
            Int: 每隔多少分钟保存一个checkpoint文件。

    .. py:method:: save_checkpoint_seconds
        :property:

        获取每隔多少秒保存一次checkpoint文件。

        返回：
            Int: 每隔多少秒保存一次checkpoint文件。

    .. py:method:: save_checkpoint_steps
        :property:

        获取每隔多少个step保存一次checkpoint文件。

        返回：
            Int: 每隔多少个step保存一次checkpoint文件。

    .. py:method:: saved_network
        :property:

        获取需要保存的网络。

        返回：
            Cell: 需要保存的网络。
