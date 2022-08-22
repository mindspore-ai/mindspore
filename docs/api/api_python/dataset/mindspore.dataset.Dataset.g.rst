.. py:method:: set_dynamic_columns(columns=None)

    设置数据集的动态shape信息，需要在定义好完整的数据处理管道后进行设置。

    参数：
        - **columns** (dict) - 包含数据集中每列shape信息的字典。shape[i]为 `None` 表示shape[i]的数据长度是动态的。

.. py:method:: shuffle(buffer_size)

    使用以下策略混洗此数据集的行：

    1. 生成一个混洗缓冲区包含 `buffer_size` 条数据行。

    2. 从混洗缓冲区中随机选择一个数据行，传递给下一个操作。

    3. 从上一个操作获取下一个数据行（如果有的话），并将其放入混洗缓冲区中。

    4. 重复步骤2和3，直到混洗缓冲区中没有数据行为止。

    在第一个epoch中可以通过 `dataset.config.set_seed` 来设置随机种子，在随后的每个epoch，种子都会被设置成一个新产生的随机值。

    参数：
        - **buffer_size** (int) - 用于混洗的缓冲区大小（必须大于1）。将 `buffer_size` 设置为数据集大小将进行全局混洗。

    返回：
        ShuffleDataset，混洗后的数据集对象。

    异常：
        - **RuntimeError** - 混洗前存在通过 `dataset.sync_wait` 进行同步操作。

.. py:method:: skip(count)

    跳过此数据集对象的前 `count` 条数据。

    参数：
        - **count** (int) - 要跳过数据的条数。

    返回：
        SkipDataset，跳过指定条数据后的数据集对象。

.. py:method:: split(sizes, randomize=True)

    将数据集拆分为多个不重叠的子数据集。

    参数：
        - **sizes** (Union[list[int], list[float]]) - 如果指定了一列整数[s1, s2, …, sn]，数据集将被拆分为n个大小为s1、s2、...、sn的数据集。如果所有输入大小的总和不等于原始数据集大小，则报错。如果指定了一列浮点数[f1, f2, …, fn]，则所有浮点数必须介于0和1之间，并且总和必须为1，否则报错。数据集将被拆分为n个大小为round(f1*K)、round(f2*K)、...、round(fn*K)的数据集，其中K是原始数据集的大小。

          如果round四舍五入计算后：

          - 任何子数据集的的大小等于0，都将发生错误。
          - 如果子数据集大小的总和小于K，K - sigma(round(fi * k))的值将添加到第一个子数据集，sigma为求和操作。
          - 如果子数据集大小的总和大于K，sigma(round(fi * K)) - K的值将从第一个足够大的子数据集中删除，且删除后的子数据集大小至少大于1。

        - **randomize** (bool, 可选) - 确定是否随机拆分数据，默认值：True，数据集将被随机拆分。否则将按顺序拆分为多个不重叠的子数据集。

    .. note::
        1. 如果进行拆分操作的数据集对象为MappableDataset类型，则将自动调用一个优化后的split操作。
        2. 如果进行split操作，则不应对数据集对象进行分片操作（如指定num_shards或使用DistributerSampler）。相反，如果创建一个DistributerSampler，并在split操作拆分后的子数据集对象上进行分片操作，强烈建议在每个子数据集上设置相同的种子，否则每个分片可能不是同一个子数据集的一部分（请参见示例）。
        3. 强烈建议不要对数据集进行混洗，而是使用随机化（randomize=True）。对数据集进行混洗的结果具有不确定性，每个拆分后的子数据集中的数据在每个epoch可能都不同。

    异常：
        - **RuntimeError** - 数据集对象不支持 `get_dataset_size` 或者 `get_dataset_size` 返回None。
        - **RuntimeError** - `sizes` 是list[int]，并且 `sizes` 中所有元素的总和不等于数据集大小。
        - **RuntimeError** - `sizes` 是list[float]，并且计算后存在大小为0的拆分子数据集。
        - **RuntimeError** - 数据集对象在调用拆分之前已进行分片。
        - **ValueError** - `sizes` 是list[float]，且并非所有float数值都在0和1之间，或者float数值的总和不等于1。

    返回：
        tuple(Dataset)，split操作后子数据集对象的元组。

.. py:method:: sync_update(condition_name, num_batch=None, data=None)

    释放阻塞条件并使用给定数据触发回调函数。

    参数：
        - **condition_name** (str) - 用于触发发送下一个数据行的条件名称。
        - **num_batch** (Union[int, None]) - 释放的batch（row）数。当 `num_batch` 为None时，将默认为 `sync_wait`  操作指定的值，默认值：None。
        - **data** (Any) - 用户自定义传递给回调函数的数据，默认值：None。

.. py:method:: sync_wait(condition_name, num_batch=1, callback=None)

    为同步操作在数据集对象上添加阻塞条件。

    参数：
        - **condition_name** (str) - 用于触发发送下一行数据的条件名称。
        - **num_batch** (int) - 每个epoch开始时无阻塞的batch数。
        - **callback** (function) - `sync_update` 操作中将调用的回调函数。

    返回：
        SyncWaitDataset，添加了阻塞条件的数据集对象。

    异常：
        - **RuntimeError** - 条件名称已存在。

.. py:method:: take(count=-1)

    从数据集中获取最多 `count` 的元素。

    .. note::
        1. 如果 `count` 大于数据集中的数据条数或等于-1，则取数据集中的所有数据。
        2. take和batch操作顺序很重要，如果take在batch操作之前，则取给定条数，否则取给定batch数。

    参数：
        - **count** (int, 可选) - 要从数据集对象中获取的数据条数，默认值：-1，获取所有数据。

    返回：
        TakeDataset，take操作后的数据集对象。

.. py:method:: to_device(send_epoch_end=True, create_data_info_queue=False)

    将数据从CPU传输到GPU、Ascend或其他设备。

    参数：
        - **send_epoch_end** (bool, 可选) - 是否将epoch结束符 `end_of_sequence` 发送到设备，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建存储数据类型和shape的队列，默认值：False。

    .. note::
        该接口在将来会被删除或不可见。建议使用 `device_queue` 接口。
        如果设备为Ascend，则逐个传输数据。每次数据传输的限制为256M。

    返回：
        TransferDataset，用于传输的数据集对象。

    异常：
        - **RuntimeError** - 如果提供了分布式训练的文件路径但读取失败。

.. py:method:: to_json(filename='')

    将数据处理管道序列化为JSON字符串，如果提供了文件名，则转储到文件中。

    参数：
        - **filename** (str) - 保存JSON文件的路径（包含文件名）。

    返回：
        str，数据处理管道序列化后的JSON字符串。
