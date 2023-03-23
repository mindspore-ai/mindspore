mindspore.dataset.Graph
=======================

.. py:class:: mindspore.dataset.Graph(edges, node_feat=None, edge_feat=None, graph_feat=None, node_type=None, edge_type=None, num_parallel_workers=None, working_mode='local', hostname='127.0.0.1', port=50051, num_client=1, auto_shutdown=True)

    主要用于存储图的结构信息和图特征属性，并提供图采样等能力。

    该接口支持输入表示节点、边及其特征的NumPy数组，来进行图初始化。如果 `working_mode` 是默认的 `local` 模式，则不需要指定 `working_mode`、`hostname`、`port`、`num_client`、`auto_shutdown` 等输入参数。

    参数：
        - **edges** (Union[list, numpy.ndarray]) - 以COO格式表示的边，shape为 [2, num_edges]。
        - **node_feat** (dict, 可选) - 节点的特征，输入数据格式应该是dict，其中key表示特征的类型，用字符串表示，比如 'weight' 等；value应该是shape为 [num_nodes, num_node_features] 的NumPy数组。
        - **edge_feat** (dict, 可选) - 边的特征，输入数据格式应该是dict，其中key表示特征的类型，用字符串表示，比如 'weight' 等；value应该是shape为 [num_edges, num_edge_features] 的NumPy数组。
        - **graph_feat** (dict, 可选) - 附加特征，不能分配给 `node_feat` 或者 `edge_feat`，输入数据格式应该是dict，key是特征的类型，用字符串表示; value应该是NumPy数组，其shape可以不受限制。
        - **node_type** (Union[list, numpy.ndarray], 可选) - 节点的类型，每个元素都是字符串，表示每个节点的类型。如果未提供，则每个节点的默认类型为“0”。
        - **edge_type** (Union[list, numpy.ndarray], 可选) - 边的类型，每个元素都是字符串，表示每条边的类型。如果未提供，则每条边的默认类型为“0”。
        - **num_parallel_workers** (int, 可选) - 读取数据的工作线程数。默认值：None，使用 `mindspore.dataset.config` 中配置的线程数。
        - **working_mode** (str, 可选) - 设置工作模式，目前支持 'local'/'client'/'server'。默认值：'local'。

          - **local**：用于非分布式训练场景。
          - **client**：用于分布式训练场景。客户端不加载数据，而是从服务器获取数据。
          - **server**：用于分布式训练场景。服务器加载数据并可供客户端使用。

        - **hostname** (str, 可选) - 图数据集服务器的主机名。该参数仅在工作模式设置为 'client' 或 'server' 时有效。默认值：'127.0.0.1'。
        - **port** (int, 可选) - 图数据服务器的端口，取值范围为1024-65535。此参数仅当工作模式设置为 'client' 或 'server' 时有效。默认值：50051。
        - **num_client** (int, 可选) - 期望连接到服务器的最大客户端数。服务器将根据该参数分配资源。该参数仅在工作模式设置为 'server' 时有效。默认值：1。
        - **auto_shutdown** (bool, 可选) - 当工作模式设置为 'server' 时有效。当连接的客户端数量达到 `num_client` ，且没有客户端正在连接时，服务器将自动退出。默认值：True。

    异常：
        - **TypeError** - 如果 `edges` 不是list或NumPy array类型。
        - **TypeError** - 如果提供了 `node_feat` 但不是dict类型, 或者dict中的key不是string类型, 或者dict中的value不是NumPy array类型。
        - **TypeError** - 如果提供了 `edge_feat` 但不是dict类型, 或者dict中的key不是string类型, 或者dict中的value不是NumPy array类型。
        - **TypeError** - 如果提供了 `graph_feat` 但不是dict类型, 或者dict中的key不是string类型, 或者dict中的value不是NumPy array类型。
        - **TypeError** - 如果提供了 `node_type` 但不是list或NumPy array类型。
        - **TypeError** - 如果提供了 `edge_type` 但不是list或 NumPy array类型。
        - **ValueError** - `num_parallel_workers` 参数超过系统最大线程数。
        - **ValueError** - `working_mode` 参数取值不为 'local', 'client' 或 'server'。
        - **TypeError** - `hostname` 参数类型错误。
        - **ValueError** - `port` 参数不在范围[1024, 65535]内。
        - **ValueError** - `num_client` 参数不在范围[1, 255]内。

    .. py:method:: get_all_edges(edge_type)

        获取图的所有边。

        参数：
            - **edge_type** (str) - 指定边的类型。默认值：'0'。

        返回：
            numpy.ndarray，包含边的数组。

        异常：
            - **TypeError** - 参数 `edge_type` 的类型不是string类型。

    .. py:method:: get_all_neighbors(node_list, neighbor_type, output_format=OutputFormat.NORMAL)

        获取 `node_list` 所有节点的相邻节点，以 `neighbor_type` 类型返回。格式的定义参见以下示例。1表示两个节点之间连接，0表示不连接。

        .. list-table:: 邻接矩阵
            :widths: 20 20 20 20 20
            :header-rows: 1

            * -
              - 0
              - 1
              - 2
              - 3
            * - 0
              - 0
              - 1
              - 0
              - 0
            * - 1
              - 0
              - 0
              - 1
              - 0
            * - 2
              - 1
              - 0
              - 0
              - 1
            * - 3
              - 1
              - 0
              - 0
              - 0

        .. list-table:: 普通格式
            :widths: 20 20 20 20 20
            :header-rows: 1

            * - src
              - 0
              - 1
              - 2
              - 3
            * - dst_0
              - 1
              - 2
              - 0
              - 1
            * - dst_1
              - -1
              - -1
              - 3
              - -1

        .. list-table:: COO格式
            :widths: 20 20 20 20 20 20
            :header-rows: 1

            * - src
              - 0
              - 1
              - 2
              - 2
              - 3
            * - dst
              - 1
              - 2
              - 0
              - 3
              - 1

        .. list-table:: CSR格式
            :widths: 40 20 20 20 20 20
            :header-rows: 1

            * - offsetTable
              - 0
              - 1
              - 2
              - 4
              -
            * - dstTable
              - 1
              - 2
              - 0
              - 3
              - 1

        参数：
            - **node_list** (Union[list, numpy.ndarray]) - 给定的节点列表。
            - **neighbor_type** (str) - 指定相邻节点的类型。
            - **output_format** (OutputFormat, 可选) - 输出存储格式。默认值：mindspore.dataset.OutputFormat.NORMAL，取值范围：[OutputFormat.NORMAL, OutputFormat.COO, OutputFormat.CSR]。

        返回：
            对于普通格式或COO格式，将返回numpy.ndarray类型的数组表示相邻节点。如果指定了CSR格式，将返回两个numpy.ndarray数组，第一个表示偏移表，第二个表示相邻节点。

        异常：
            - **TypeError** - 参数 `node_list` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `neighbor_type` 的类型不是string类型。

    .. py:method:: get_all_nodes(node_type)

        获取图中的所有节点。

        参数：
            - **node_type** (str) - 指定节点的类型。默认值：'0'。

        返回：
            numpy.ndarray，包含节点的数组。
        
        异常：
            - **TypeError** - 参数 `node_type` 的类型不是string类型。

    .. py:method:: get_edge_feature(edge_list, feature_types)

        获取 `edge_list` 列表中边的特征，以 `feature_types` 类型返回。

        参数：
            - **edge_list** (Union[list, numpy.ndarray]) - 包含边的列表。
            - **feature_types** (Union[list, numpy.ndarray]) - 包含给定特征类型的列表，列表中每个元素是string类型。

        返回：
            numpy.ndarray，包含特征的数组。

        异常：
            - **TypeError** - 参数 `edge_list` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `feature_types` 的类型不为列表或numpy.ndarray。

    .. py:method:: get_edges_from_nodes(node_list)

        从节点获取边。

        参数：
            - **node_list** (Union[list[tuple], numpy.ndarray]) - 含一个或多个图节点ID对的列表。

        返回：
            numpy.ndarray，含一个或多个边ID的数组。

        异常：
            - **TypeError** - 参数 `edge_list` 的类型不为列表或numpy.ndarray。

    .. py:method:: get_graph_feature(feature_types)

        依据给定的 `feature_types` 获取存储在Graph中对应的特征。

        参数：
            - **feature_types** (Union[list, numpy.ndarray]) - 包含给定特征类型的列表，列表中每个元素是string类型。

        返回：
            numpy.ndarray，包含特征的数组。

        异常：
            - **TypeError** - 参数 `feature_types` 的类型不为列表或numpy.ndarray。

    .. py:method:: get_neg_sampled_neighbors(node_list, neg_neighbor_num, neg_neighbor_type)

        获取 `node_list` 列表中节所有点的负样本相邻节点，以 `neg_neighbor_type` 类型返回。

        参数：
            - **node_list** (Union[list, numpy.ndarray]) - 包含节点的列表。
            - **neg_neighbor_num** (int) - 采样的相邻节点数量。
            - **neg_neighbor_type** (str) - 指定负样本相邻节点的类型。

        返回：
            numpy.ndarray，包含相邻节点的数组。

        异常：
            - **TypeError** - 参数 `node_list` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `neg_neighbor_num` 的类型不为整型。
            - **TypeError** - 参数 `neg_neighbor_type` 的类型不是string类型。

    .. py:method:: get_node_feature(node_list, feature_types)

        获取 `node_list` 中节点的特征，以 `feature_types` 类型返回。

        参数：
            - **node_list** (Union[list, numpy.ndarray]) - 包含节点的列表。
            - **feature_types** (Union[list, numpy.ndarray]) - 指定特征的类型，类型列表中每个元素应该是string类型。

        返回：
            numpy.ndarray，包含特征的数组。

        异常：
            - **TypeError** - 参数 `node_list` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `feature_types` 的类型不为列表或numpy.ndarray。

    .. py:method:: get_nodes_from_edges(edge_list)

        从图中的边获取节点。

        参数：
            - **edge_list** (Union[list, numpy.ndarray]) - 包含边的列表。

        返回：
            numpy.ndarray，包含节点的数组。

        异常：
            - **TypeError** - 参数 `edge_list` 不为列表或ndarray。

    .. py:method:: get_sampled_neighbors(node_list, neighbor_nums, neighbor_types, strategy=SamplingStrategy.RANDOM)

        获取已采样相邻节点信息。此API支持多跳相邻节点采样。即将上一次采样结果作为下一跳采样的输入。最多允许6跳。采样结果平铺成列表，格式为[input node, 1-hop sampling result, 2-hop samling result ...]。

        参数：
            - **node_list** (Union[list, numpy.ndarray]) - 包含节点的列表。
            - **neighbor_nums** (Union[list, numpy.ndarray]) - 每跳采样的相邻节点数。
            - **neighbor_types** (Union[list, numpy.ndarray]) - 每跳采样的相邻节点类型，列表或数组中每个元素都应该是字符串类型。
            - **strategy** (SamplingStrategy, 可选) - 采样策略。默认值：mindspore.dataset.SamplingStrategy.RANDOM。取值范围：[SamplingStrategy.RANDOM, SamplingStrategy.EDGE_WEIGHT]。

              - **SamplingStrategy.RANDOM**：随机抽样，带放回采样。
              - **SamplingStrategy.EDGE_WEIGHT**：以边缘权重为概率进行采样。

        返回：
            numpy.ndarray，包含相邻节点的数组。

        异常：
            - **TypeError** - 参数 `node_list` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `neighbor_nums` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `neighbor_types`  的类型不为列表或numpy.ndarray。

    .. py:method:: graph_info()

        获取图的元信息，包括节点数、节点类型、节点特征信息、边数、边类型、边特征信息。

        返回：
            dict，图的元信息。键为 `node_num` 、 `node_type` 、 `node_feature_type` 、 `edge_num` 、 `edge_type` 、`edge_feature_type` 和 `graph_feature_type`。

    .. py:method:: random_walk(target_nodes, meta_path, step_home_param=1.0, step_away_param=1.0, default_node=-1)

        在节点中的随机游走。

        参数：
            - **target_nodes** (list[int]) - 随机游走中的起始节点列表。
            - **meta_path** (list[int]) - 每个步长的节点类型。
            - **step_home_param** (float, 可选) - 返回 `node2vec算法 <https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf>`_ 中的超参。默认值：1.0。
            - **step_away_param** (float, 可选) - `node2vec算法 <https://www.kdd.org/kdd2016/papers/files/rfp0218-groverA.pdf>`_ 中的in和out超参。默认值：1.0。
            - **default_node** (int, 可选) - 如果找不到更多相邻节点，则为默认节点。默认值：-1，表示不给定节点。

        返回：
            numpy.ndarray，包含节点的数组。

        异常：
            - **TypeError** - 参数 `target_nodes` 的类型不为列表或numpy.ndarray。
            - **TypeError** - 参数 `meta_path` 的类型不为列表或numpy.ndarray。
