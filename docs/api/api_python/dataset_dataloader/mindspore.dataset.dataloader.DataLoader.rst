mindspore.dataset.dataloader.DataLoader
=======================================

.. py:class:: mindspore.dataset.dataloader.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, \
    batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0.0, \
    worker_init_fn=None, multiprocessing_context=None, generator=None, *, prefetch_factor=None, \
    persistent_workers=False, in_order=True)

    数据加载器为给定的数据集提供迭代器。

    它支持可随机访问和可迭代类型的数据集，支持单进程或多进程数据加载。

    参数：
        - **dataset** (Dataset) - 要从中加载数据的数据集。
        - **batch_size** (Union[int, None], 可选) - 每个 mini-batch 的样本数量。如果为 ``None`` ，则不进行批处理。
          默认值： ``1`` 。
        - **shuffle** (Union[bool, None], 可选) - 是否打乱数据集。默认值： ``None`` ，不打乱。
        - **sampler** (Union[Sampler, Iterable, None], 可选) - 要使用的采样器。默认值： ``None`` ，如果 `shuffle` 为
          ``False`` 则使用 :class:`~mindspore.dataset.dataloader.SequentialSampler` ，否则使用
          :class:`~mindspore.dataset.dataloader.RandomSampler` 。
        - **batch_sampler** (Union[Sampler[List], Iterable[List], None], 可选) - 要使用的批采样器。默认值： ``None`` ，
          如果 `batch_size` 不为 ``None`` 则生成内置 :class:`~mindspore.dataset.dataloader.BatchSampler` 。
        - **num_workers** (int, 可选) - 执行数据加载的工作进程数。默认值： ``0`` ，使用主进程加载。
        - **collate_fn** (Union[_CollateFnType, None], 可选) - 要使用的整理函数。默认值： ``None`` ，使用默认整理函数。
        - **pin_memory** (bool, 可选) - 是否将数据拷贝到锁页内存。默认值： ``False`` 。
        - **drop_last** (bool, 可选) - 是否丢弃最后一个不完整的 Batch 。默认值： ``False`` 。
        - **timeout** (float, 可选) - 等待工作进程处理数据的超时时间。默认值： ``0.0`` ，永久等待。
        - **worker_init_fn** (Union[Callable[[int], None], None], 可选) - 要使用的工作进程初始化函数。
          默认值： ``None`` ，不执行任何操作。
        - **multiprocessing_context** (Union[multiprocessing.context.BaseContext, str, None], 可选) - 要使用的多进程上下文。
          默认值： ``None`` ，使用 :mod:`mindspore.multiprocessing` 。
        - **generator** (Union[numpy.random.Generator, None], 可选) - 要使用的随机生成器。默认值： ``None`` ，使用默认生成器。

    关键字参数：
        - **prefetch_factor** (Union[int, None], 可选) - 工作进程预取样本数。
          默认值： ``None`` ，当 `num_workers` 大于 ``0`` 时使用 ``2`` 。
        - **persistent_workers** (bool, 可选) - 是否在迭代数据集完成后保持工作进程存活。默认值： ``False`` 。
        - **in_order** (bool, 可选) - 在多进程加载时是否保持样本顺序。默认值： ``True`` 。
