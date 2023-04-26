# MindSpore Release Notes

[View English](./RELEASE.md)

## MindSpore 2.0.0-rc1 Release Notes

### 主要特性和增强

#### FrontEnd

- [BETA] 静态图模式下，函数及类方法支持"return None"、"return"、无"return"语法。
- [BETA] 静态图模式下，支持返回list类型对象。
- [BETA] 静态图模式下，变量条件时，支持"raise"语法。
- [STABLE] 函数式调用支持数据下沉模式。
- [BETA] nn下新增Transformer层，提供更加易用的Transformer API，无需定义batch_size，支持动态seq_length。

#### DataSet

- [STABLE] Ascend环境下，数据下沉模式超时等待时间调整，默认调整到1900s，以解决数据下沉模式时因环境资源竞争、计算量大等因素容易导致GetNext算子等待超时的问题。
- [STABLE] MindRecord提供Schema、样本数查询接口，并提供多进程并行写入功能，允许用户更快生成MindRecord数据文件。
- [STABLE] Dataset流水线支持处理任意Python对象，用法参考[数据pipeline支持Python对象](https://www.mindspore.cn/tutorials/zh-CN/r2.0/advanced/dataset/python_objects.html)。

#### AutoParallel

- [STABLE] 策略保存时支持保存完整策略。
- [STABLE] 支持Conv3D/MaxPool3D/AvgPool3D分布式算子。
- [STABLE] 支持PyNative+shard算子级并行+优化器并行：并行表达和Model进行解耦，提供基础的并行表达能力。
- [STABLE] 支持图模式算子级并行+优化器并行：并行表达和Model进行解耦，提供基础的并行表达能力。
- [BETA] 支持自定义分布式图切分，提升分布式训练的灵活性。

#### Runtime

- [STABLE] 控制流支持子图下沉。
- [STABLE] 支持CUDA 11.6。
- [STABLE] 支持List/Tuple/Scalar类型算子的算子选择和执行，配套Python原生表达。
- [STABLE] 硬件不支持的算子自动选择CPU算子。
- [STABLE] 支持子图内部异构执行。

#### Ascend

- [STABLE] 支持CANN溢出检测新方案和HCCL运行态溢出检测。
- [STABLE] 支持集合通信算子dump功能。

#### Profiler

- [STABLE] 丰富Profiler采集项配置，用户可以更细度地采集性能数据。

#### Dump

- [BETA] 单卡PyNatvie模式支持算子溢出检测。
- [BETA] Graph模式支持hccl算子dump。

### API变更

- [STABLE] 新增计算类API，如：MaxUnpool、ReplicationPad、GaussianNLLLoss等。
  详情请参考：<https://www.mindspore.cn/docs/zh-CN/r2.0/api_python/mindspore.html>。
- [STABLE] 扩展存量API功能，如：AvgPool、pad、norm、interplate等。

#### 算子

- [BETA] `mindspore.ops.AdaptiveAvgPool3D` 新增算子原语。
- [BETA] `mindspore.ops.AffineGrid` 新增算子原语。
- [BETA] `mindspore.ops.Angle` 新增算子原语。
- [BETA] `mindspore.ops.BartlettWindow` 新增算子原语。
- [BETA] `mindspore.ops.Bernoulli` 新增算子原语。
- [BETA] `mindspore.ops.BesselI0` 新增算子原语。
- [BETA] `mindspore.ops.BesselI1` 新增算子原语。
- [BETA] `mindspore.ops.BesselJ0` 新增算子原语。
- [BETA] `mindspore.ops.BesselJ1` 新增算子原语。
- [BETA] `mindspore.ops.BesselK0` 新增算子原语。
- [BETA] `mindspore.ops.BesselK0e` 新增算子原语。
- [BETA] `mindspore.ops.BesselK1` 新增算子原语。
- [BETA] `mindspore.ops.BesselK1e` 新增算子原语。
- [BETA] `mindspore.ops.BesselY0` 新增算子原语。
- [BETA] `mindspore.ops.BesselY1` 新增算子原语。
- [BETA] `mindspore.ops.Bincount` 新增算子原语。
- [BETA] `mindspore.ops.BlackmanWindow` 新增算子原语。
- [BETA] `mindspore.ops.ChannelShuffle` 新增算子原语。
- [BETA] `mindspore.ops.Cholesky` 新增算子原语。
- [BETA] `mindspore.ops.Col2Im` 新增算子原语。
- [BETA] `mindspore.ops.Complex` 新增算子原语。
- [BETA] `mindspore.ops.ComplexAbs` 新增算子原语。
- [BETA] `mindspore.ops.Cross` 新增算子原语。
- [BETA] `mindspore.ops.CTCLossV2` 新增算子原语。
- [BETA] `mindspore.ops.Cummin` 新增算子原语。
- [BETA] `mindspore.ops.Diag` 新增算子原语。
- [BETA] `mindspore.ops.Digamma` 新增算子原语。
- [BETA] `mindspore.ops.Eig` 新增算子原语。
- [BETA] `mindspore.ops.Expand` 新增算子原语。
- [BETA] `mindspore.ops.Fmax` 新增算子原语。
- [BETA] `mindspore.ops.Gcd` 新增算子原语。
- [BETA] `mindspore.ops.Geqrf` 新增算子原语。
- [BETA] `mindspore.ops.GLU` 新增算子原语。
- [BETA] `mindspore.ops.GridSampler2D` 新增算子原语。
- [BETA] `mindspore.ops.GridSampler3D` 新增算子原语。
- [BETA] `mindspore.ops.HammingWindow` 新增算子原语。
- [BETA] `mindspore.ops.Heaviside` 新增算子原语。
- [BETA] `mindspore.ops.Hypot` 新增算子原语。
- [BETA] `mindspore.ops.Igamma` 新增算子原语。
- [BETA] `mindspore.ops.IndexFill` 新增算子原语。
- [BETA] `mindspore.ops.InplaceIndexAdd` 新增算子原语。
- [BETA] `mindspore.ops.InplaceUpdateV2` 新增算子原语。
- [BETA] `mindspore.ops.Lcm` 新增算子原语。
- [BETA] `mindspore.ops.LeftShift` 新增算子原语。
- [BETA] `mindspore.ops.LogicalXor` 新增算子原语。
- [BETA] `mindspore.ops.Logit` 新增算子原语。
- [BETA] `mindspore.ops.LogSpace` 新增算子原语。
- [BETA] `mindspore.ops.LuUnpack` 新增算子原语。
- [BETA] `mindspore.ops.MatrixDiagPartV3` 新增算子原语。
- [BETA] `mindspore.ops.MatrixDiagV3` 新增算子原语。
- [BETA] `mindspore.ops.MatrixSetDiagV3` 新增算子原语。
- [BETA] `mindspore.ops.MaxPool3DWithArgmax` 新增算子原语。
- [BETA] `mindspore.ops.MaxUnpool2D` 新增算子原语。
- [BETA] `mindspore.ops.MaxUnpool3D` 新增算子原语。
- [BETA] `mindspore.ops.MultiMarginLoss` 新增算子原语。
- [BETA] `mindspore.ops.MultinomialWithReplacement` 新增算子原语。
- [BETA] `mindspore.ops.Mvlgamma` 新增算子原语。
- [BETA] `mindspore.ops.NanToNum` 新增算子原语。
- [BETA] `mindspore.ops.NextAfter` 新增算子原语。
- [BETA] `mindspore.ops.Orgqr` 新增算子原语。
- [BETA] `mindspore.ops.Polygamma` 新增算子原语。
- [BETA] `mindspore.ops.Qr` 新增算子原语。
- [BETA] `mindspore.ops.ResizeBilinearV2` 新增算子原语。
- [BETA] `mindspore.ops.RightShift` 新增算子原语。
- [BETA] `mindspore.ops.ScatterNdDiv` 新增算子原语。
- [BETA] `mindspore.ops.ScatterNdMul` 新增算子原语。
- [BETA] `mindspore.ops.SearchSorted` 新增算子原语。
- [BETA] `mindspore.ops.Sinc` 新增算子原语。
- [BETA] `mindspore.ops.Trace` 新增算子原语。
- [BETA] `mindspore.ops.Tril` 新增算子原语。
- [BETA] `mindspore.ops.TrilIndices` 新增算子原语。
- [BETA] `mindspore.ops.TriuIndices` 新增算子原语。
- [BETA] `mindspore.ops.UniqueConsecutive` 新增算子原语。
- [STABLE] `mindspore.ops.Cummax` 新增算子原语。
- [STABLE] `mindspore.ops.FillV2` 新增算子原语。
- [STABLE] `mindspore.ops.IsClose` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixSolve` 新增算子原语。
- [STABLE] `mindspore.ops.Median` 新增算子原语。
- [STABLE] `mindspore.ops.MultilabelMarginLoss` 新增算子原语。
- [STABLE] `mindspore.ops.NonZero` 新增算子原语。
- [STABLE] `mindspore.ops.Pdist` 新增算子原语。
- [STABLE] `mindspore.ops.Polar` 新增算子原语。
- [STABLE] `mindspore.ops.RandomGamma` 新增算子原语。
- [STABLE] `mindspore.ops.RandomPoisson` 新增算子原语。
- [STABLE] `mindspore.ops.RandomShuffle` 新增算子原语。
- [STABLE] `mindspore.ops.Renorm` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMax` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMin` 新增算子原语。
- [STABLE] `mindspore.ops.Svd` 新增算子原语。
- [STABLE] `mindspore.ops.TripletMarginLoss` 新增算子原语。

#### 删除接口

- `mindspore.compression`特性在MindSpore 1.8版本已经废弃，在当前版本被删除。用户可以使用[昇思金箍棒](https://gitee.com/mindspore/golden-stick)作为`mindspore.compression`的替代品来实现MindSpore中的量化感知训练算法。
- `mindspore.dataset.close_pool`、`mindspore.dataset.to_device`、`mindspore.dataset.set_dynamic_columns` 接口在之前版本已废弃，当前版本正式删除。

#### 非兼容性接口变更

- 接口名称：mindspore.set_context(mode=PYNATIVE_MODE)

  变更内容：默认由GRAPH_MODE改为PYNATIVE_MODE。

  说明：原有使用方式若未设置运行模式，该变更会影响性能，需要额外设置图模式，则使用以下方式：
  mindspore.set_context(mode=GRAPH_MODE)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.set_context(mode=GRAPH_MODE)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.set_context(mode=PYNATIVE_MODE)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.train.Model.train

  变更内容：dataset_sink_mode 默认值由True改为False。

  说明：原有使用方式若未设置dataset_sink_mode，该变更会影响性能，需要额外设置数据下沉运行模式，则使用以下方式：
  Model.train(dataset_sink_mode=True)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  Model.train(dataset_sink_mode=True)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  Model.train(dataset_sink_mode=False)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.export

  变更内容：参数file_format由"AIR"改为不指定默认值。

  说明：原有使用方式若未设置file_format，需要额外设置file_format，则使用以下方式：
  mindspore.export(net, *inputs, file_name, file_format="AIR", **kwargs)。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.export(net, *inputs, file_name,
                   file_format="AIR", **kwargs)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.export(net, *inputs, file_name,
                   file_format, **kwargs)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.norm

  变更内容：扩展ord参数功能，支持多种形式。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.norm(input_x, axis, p=2, keep_dims=False, epsilon=1e-12)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                            [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, [0, 1], p=2)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.norm(A, ord=None, dim=None, keepdim=False, *, dtype=None)
  >>> # 举例:
  >>> input = Tensor(np.array([[[1.0, 2.0], [3.0, 4.0]],
  ...                            [[5.0, 6.0], [7.0, 8.0]]]).astype(np.float32))
  >>> output = ops.norm(input, ord=2, dim=(0, 1))
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.norm

  变更内容：扩展ord参数功能，支持多种形式。

  说明：参考ops.norm例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  Tensor.norm(axis, p=2, keep_dims=False, epsilon=1e-12)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  Tensor.norm(ord=None, dim=None, keepdim=False, *, dtype=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout

  变更内容：删除seed0、seed1参数，新增参数seed=None。由返回Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout(x, p=0.5, seed0=0, seed1=0)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output, mask = dropout(x, p=0.5)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout(input, p=0.5, training=True, seed=None)
  >>> # 举例:
  >>> input = Tensor(((20, 16), (50, 50)),
  ...                mindspore.float32)
  >>> output = ops.dropout(input, p=0.5，training=True)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout2d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout2d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout2d(input, 0.5)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout2d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout2d(input, 0.5, training=True)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.dropout3d

  变更内容：返回值从Tensor和掩码改为只返回Tensor，新增入参training=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout3d(x, p=0.5)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output, mask = dropout3d(input, 0.5)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.dropout3d(input, p=0.5, training=True)
  >>> # 举例:
  >>> input = Tensor(np.ones([2, 1, 2, 3]),
  ...                mindspore.float32)
  >>> output = ops.dropout3d(input, 0.5, training=True)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.std

  变更内容：接口重构，接口使用方式更符合用户使用习惯。

  说明：原有unbiased如果已显示设置，采用以下替代方案：
  ddof=0替代unbiased=False，ddof=1替代unbiased=True。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.std(input_x, axis=(), unbiased=True, keep_dims=False)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.std(input, axis=None, ddof=0, keepdims=False)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.load_param_into_net

  变更内容：新增ckpt中未加载的参数作为返回值。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  net_param = load_param_into_net()
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  net_param, ckpt_param = load_param_into_net()
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.nn.BCELoss

  变更内容：`reduction` 默认值由'none'变为'mean'。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  BCELoss(weight=None, reduction='none')
  >>> # 举例:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight, reduction='mean')
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  BCELoss(weight=None, reduction='mean')
  >>> # 举例:
  >>> weight = Tensor(np.array([[1.0, 2.0, 3.0],
  ...                           [4.0, 3.3, 2.2]]),
  ...                 mindspore.float32)
  >>> loss = nn.BCELoss(weight=weight)
  >>> logits = Tensor(np.array([[0.1, 0.2, 0.3],
  ...                           [0.5, 0.7, 0.9]]),
  ...                 mindspore.float32)
  >>> labels = Tensor(np.array([[0, 1, 0], [0, 0, 1]]),
  ...                 mindspore.float32)
  >>> output = loss(logits, labels)
  >>> print(output)
  >>> 1.8952923
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整第2个和第3个参数的顺序，修改并扩展split_size_or_sections功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.split(input_x, axis=0, output_num=1)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, axis=1, output_num=4)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.split(tensor, split_size_or_sections, axis=0)
  >>> # 举例:
  >>> input = Tensor(np.array([[1, 1, 1, 1], [2, 2, 2, 2]]),
  ...                mindspore.int32)
  >>> output = ops.split(input, split_size_or_sections=1, axis=1)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.Tensor.split

  变更内容：接口重构，接口使用方式更符合用户使用习惯，调整两个参数的位置，修改并扩展split_size_or_sections功能。

  说明：参考ops.split例子。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  Tensor.split(axis=0, output_num=1)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  Tensor.split(split_size_or_sections, axis=0)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.pad

  变更内容：修改参数名paddings为padding，添加mode和value功能。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.pad(input_x, paddings)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = ((1, 2), (2, 1))
  >>> output = ops.pad(input_x, paddings)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.pad(input_x, padding, mode='constant', value=None)
  >>> # 举例:
  >>> input_x = Tensor(np.array([[-0.1, 0.3, 3.6],
  ...                            [0.4, 0.5, -3.2]]),
  ...                  mindspore.float32)
  >>> paddings = (2, 1, 1, 2)
  >>> output = ops.pad(input_x, paddings)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.meshgrid

  变更内容：入参由inputs改为*input。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.meshgrid(inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid((x, y, z), indexing='xy')
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.meshgrid(*inputs, indexing='xy')
  >>> # 举例:
  >>> x = Tensor(np.array([1, 2, 3, 4]).astype(np.int32))
  >>> y = Tensor(np.array([5, 6, 7]).astype(np.int32))
  >>> z = Tensor(np.array([8, 9, 0, 1, 2]).astype(np.int32))
  >>> output = ops.meshgrid(x, y, z, indexing='xy')
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.max

  变更内容：返回值调换顺序，由：“下标，最大值”改为“最大值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.max(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.max(input)
  >>> print(index, output)
  >>> 3 0.7
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.max(input, axis=0)
  >>> print(output, index)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.min

  变更内容：返回值调换顺序，由：“下标，最小值”改为“最小值，下标”。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.min(x, axis=0, keep_dims=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> index, output = ops.min(input)
  >>> 0 0.0
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False)
  >>> # 举例:
  >>> input = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]),
  ...                mindspore.float32)
  >>> output, index = ops.min(input, keepdims=True)
  >>> 0.0 0
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.random_gamma

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.random_gamma(shape, alpha, seed=0, seed2=0)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.random_gamma(shape, alpha, seed=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.standard_laplace

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.standard_laplace(shape, seed=0, seed2=0)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.standard_laplace(shape, seed=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.standard_normal

  变更内容：删除seed2参数，seed=0改为None。框架行为统一且符合用户实际使用场景及习惯。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.standard_normal(shape, seed=0, seed2=0)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.standard_normal(shape, seed=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.bernoulli

  变更内容：seed的默认值由-1改为None。符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  ops.bernoulli(x, p=0.5, seed=-1)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  ops.bernoulli(input, p=0.5, seed=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.data_sink

  变更内容：删除steps参数，jit参数名称修改为jit_config，新增input_signature参数。增加易用性，符合用户实际使用场景。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.data_sink(fn, dataset, steps,
                      sink_size=1, jit=False)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.data_sink(fn, dataset, sink_size=1,
                      jit_config=None, input_signature=None)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.ops.conv2d

  变更内容：扩展接口功能，添加bias参数，修改参数名及参数顺序。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  conv2d(inputs, weight, pad_mode="valid",
         padding=0, stride=1, dilation=1, group=1)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  conv2d(input, weight, bias=None, stride=1,
         pad_mode="valid", padding=0, dilation=1, groups=1)
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.vision.Pad

  变更内容：调整Pad、RandomCrop、RandomCropWithBbox入参padding，当Padding输入长度为2的序列时，行为将从使用第一个值填充左/上边界，使用第二个值填充右/下边界，变为使用第一个值填充左/右边界，使用第二个值填充上/下边界。

  说明：仅使用size为2的padding参数无法兼容旧版本的效果，需显式表示（左、右、上、下）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.dataset.vision.Pad(padding=(1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  mindspore.dataset.vision.Pad(padding=(1,2,1,2))
  代表图片的左/上填充 1像素，右/下填充 2像素
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.Dataset.map

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"],
  ...                       column_order=["column_c", "column_b"])
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.map(operations=[transforms],
  ...                       input_columns=["column_a"],
  ...                       output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：删除column_order参数。因为在绝大部分的情况下，output_columns参数与column_order参数都是同一个值，不需要再传入column_order。若需要调整数据列顺序，使用mindspore.dataset.Dataset.project实现。

  说明：

  1) 在不需要改变列顺序时，直接去掉column_order参数即可。
  2) 需要指定数据列顺序时，删除column_order参数，并在后面加上一个project方法进行列变换（如下面的例子）。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.batch(batch_size=4,
  ...                         input_columns=["column_a"],
  ...                         output_columns=["column_b", "column_c"],
  ...                         column_order=["column_c", "column_b"])
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.batch(batch_size=4, input_columns=["column_a"]
  ...                         output_columns=["column_b", "column_c"])
  >>> dataset = dataset.project(["column_c", column_b"])")
  </code></pre>
  </td>
  </tr>
  </table>

- 接口名称：mindspore.dataset.Dataset.batch

  变更内容：将batch方法拆分为：batch和padded_batch两个方法。pad_info参数从batch方法移动到padded_batch方法。

  说明：如需使用pad_info参数，改用padded_batch方法。

  <table>
  <tr>
  <td style="text-align:center"> 原接口 </td> <td style="text-align:center"> v2.0.0-rc1接口 </td>
  </tr>
  <tr>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.batch(batch_size=4,
  ...                         drop_remainder=True, pad_info=...)
  </code></pre>
  <td><pre style="display: block;"><code class="language-python">
  >>> dataset = dataset.padded_batch(batch_size=4,
  ...                                drop_remainder=True, pad_info=...)
  </code></pre>
  </td>
  </tr>
  </table>

### Bug fixes

- [I66PE6] 修复 AssignSub算子异常入参导致core dump的问题。

- [I6F5E6] 修复 data_sink 方法在Ascend上执行超时的问题。

### 其它

- Windows系统支持由于还在优化中，rc版本暂不支持，将在2.0正式版本提供下载。

## MindSpore Lite 2.0.0-rc1 Release Notes

### 主要特性和增强

#### MindSpore Lite云侧推理

原MindSpore Lite版本主要面向手机、车机等边缘设备，新增云侧推理版本支持云侧多后端硬件资源的场景，支持Ascend及Nvidia GPU推理专用卡，高效利用云侧多核资源。

原通过MindSpore训练版本集成的推理方式可以变更为基于MindSpore Lite进行适配集成，具体可参考[云侧推理快速入门](https://mindspore.cn/lite/docs/zh-CN/r2.0/quick_start/one_hour_introduction_cloud.html)，如果想要保持原始集成方式可以参考[MindSpore推理FAQ](https://mindspore.cn/docs/zh-CN/r2.0/faq/inference.html)。

- [STABLE] 支持MindIR模型文件。
- [STABLE] 支持将第三方Onnx、Tensorflow、Caffe模型通过MindSpore Lite转换工具转换为MindIR模型文件。
- [STABLE] 一个发布包支持多种硬件后端：Ascend 310/310P/910、Nvidia GPU、CPU。
- [STABLE] 支持`Model`接口和`ModelParallelRunner`并行推理接口。
- [STABLE] 支持C++、Python和Java推理接口。

#### API

- 因原Python API配置参数较多、使用较复杂，因此在2.0版本针对Python API易用性进行优化，包括类构造方法、类属性的调整等，此外2.0及之后的Python API将整合到云侧推理场景，与旧版本不兼容。详细参见[Python API说明文档](https://www.mindspore.cn/lite/api/zh-CN/r2.0/mindspore_lite.html)。

### 贡献者

感谢以下人员做出的贡献：

alashkari,anzhengqi,archer2049,B.L.LAN,baihuawei,bichaoyang,BJ-WANG,Bokai Li,Brian-K,caifubi,caiyimeng,cathwong,changzherui,ChenDonYY,chenfei_mindspore,chengang,chengbin,chenhaozhe,chenjianping,chenkang,chenweifeng,chuht,chujinjin,davidanugraha,DavidFFFan,DeshiChen,douzhixing,emmmmtang,Erpim,Ethan,fangwenyi,fangzehua,fangzhou0329,fary86,fengyixing,gaoshuanglong,Gaoxiong,gaoyong10,gengdongjie,gongdaguo1,Greatpan,GuoZhibin,guozhijian,hangq,hanhuifeng,haozhang,hedongdong,Henry Shi,heterogeneous_to_backoff_2_0,huangbingjian,huanghui,huangxinjing,hujiahui8,hujingsong,huoxinyou,jachua,jiahongQian,jianghui58,jiangzhenguang,jiaorui,jiaoy1224,jijiarong,jjfeing,JoeyLin,json,JuiceZ,jxl,kairui_kou,KevinYi,kisnwang,KXiong,laiyongqiang,lanzhineng,liangchenghui,liangzelang,LiangZhibo,lianliguang,lichen,ligan,lijunbin,limingqi107,ling,linqingke,liubuyu,liuchao,liuchuting,liujunzhu,liuluobin,liutongtong9,liuyang811,lixiao,liyan2022,liyejun,liyuxia,looop5,luochao60,luojianing,luoyang,luoyuan,lyqlola,maning202007,maoyaomin,Margaret_wangrui,mayadong,MaZhiming,melody,mengyuanli,michaelzhu_70ab,Mohammad Motallebi,moran,NaCN,nomindcarry,OwenSec,panfengfeng,panshaowu,panzhihui,pkuliuliu,qinzheng,qiuzhongya,qujianwei,r1chardf1d0,Renyuan Zhang,RobinGrosman,shaojunsong,shenwei41,Soaringfish,tangdezhi_123,tanghuikang,tan-wei-cheng,TinaMengtingZhang,TronZhang,TuDouNi,VectorSL,wang_ziqi,wanghenchang,wangnan39,wangpingan,wangshaocong,wangshengnan123,wangtongyu6,weichaoran,wind-zyx,wqx,wtcheng,wujueying,wYann,XianglongZeng,xiaohanzhang,xiaotianci,xiaoyao,XinDu,xulei,xumengjuan1,xupan,xwkgch,yanghaoran,yangluhang,yangruoqi713,yangshuo,yangsijia,yangzhenzhang,yanzhenxiang2020,Yanzhi_YI,yao_yf,yefeng,yeyunpeng2020,Yi_zhang95,yide12,YijieChen,YingLai Lin,YingtongHu,youshu,yuchaojie,yuedongli,YuJianfeng,zangqx,ZengZitao,zhangbuxue,zhangdanyang,zhangdong,zhangfanghe,zhangqi,zhangqinghua,zhangyanhui,zhangyinxia,zhangyongxian,zhangzhaoju,zhanzhan,zhengzuohe,ZhidanLiu,zhixinaa,zhoufeng,zhouyaqiang0,zhuguodong,zhupuxu,zhuyuxiao,zichun_ye,zjun,zlq2020,zong_shuai,ZPaC,zuochuanyong,zyli2020,陈宇,范吉斌,冯一航,胡彬,宦晓玲,黄勇,雷元哲,李良灿,李林杰,刘崇鸣,刘力力,刘勇琪,吕浩宇,吕昱峰（Nate.River）,没有窗户的小巷,沈竞兴,十六夜,王程浩,王禹程,王振邦,徐安越,徐永飞,杨旭华,于振华,俞涵,张清华,张澍坤,张栩浩,张学同,赵英灼,周超,周洪叶,朱家兴

欢迎以任何形式对项目提供贡献！

## MindSpore 2.0.0-alpha Release Notes

### 主要特性和增强

#### PyNative

- MindSpore默认模式切换成PyNative模式。需要手动设置模式可以参考文档[计算图](https://www.mindspore.cn/tutorials/zh-CN/r2.0.0-alpha/advanced/compute_graph.html)。
- 完成动态shape执行方案重构，提升反向构图性能，支持非padding方案的动态shape网络编程，当前主要验证网络Transformer-GPU、YOLOV5-GPU、ASR-Ascend。从[models仓](https://gitee.com/mindspore/models/tree/dynamic_shape)获取Transformer-GPU和YOLOV5-GPU。Ascend后端受算子适配度限制，只支持下列算子：Add、Assign、BatchMatMul、BiasAdd、BiasAddGrad、Cast、Conv2D、Conv2DBackpropFilter、Conv2DBackpropInput、CTCLoss、Div、Dropout、DropoutDoMask、Equal、ExpandDims、Gather、GetNext、LayerNorm、LayerNormGrad、LessEqual、Load、Log、LogicalAnd、LogicalNot、LogicalOr、LogSoftmax、LogSoftmaxGrad、MatMul、Maximum、Mul、Neg、NotEqual、NPUAllocFloatStatus、NPUClearFloatStatus、OneHot、RealDiv、Reciprocal、ReduceMean、ReduceSum、ReLU、ReluGrad、Reshape、Select、Softmax、StridedSlice、Sub、Tile、Transpose、UnsortedSegmentSum、ZerosLike。其余算子未经过完整验证，请酌情使用。

#### DataSet

- TFRecordDataset API支持直接读取通过GZIP或ZLIB压缩后的TFRecord文件。
- NumpySlicesDataset API支持同时处理不同维度的数据。
- 优化错误日志信息的结构，展示更清晰的调用栈信息便于调试、定位问题。
- 修复分布式训练场景下 `mindspore.dataset.config.set_seed` 对随机种子设置不生效的问题。

#### AutoParallel

- 支持更多算子分布式能力。

  Element Wise类算子：AddN、 BitwiseAnd、 BitwiseOr、 BitwiseXor、 CumProd、 HShrink、 HSigmoid、 IsFinite、 Mish、 MulNoNan、 Rint、 SeLU、 SoftShrink、 TruncateDiv、 TruncateMod、 Xdivy Xlogy、 InplaceAdd、 InplacSub、 InplaceUpdate、 Cdist、 L2Loss、 Lerp。

  Math类算子：SquaredDifference、 Erfinv、 MaskedFill、 SplitV、 Gamma、 KLDivLoss、 LinSpace。Scatter类算子：ScatterAdd、ScatterDiv、ScatterMax、ScatterMul、ScatterNdAdd、ScatterNdSub、ScatterNdUpdate、ScatterSub、TensorScatterAdd、TensorScatterDiv、TensorScatterMax、TensorScatterMax、TensorScatterMul、TensorScatterAdd、TensorScatterUpdate。

- 增加`transform_checkpoints`和`transform_checkpoint_by_rank`接口。给定转换前后的策略文件，即可实现对分布式权重转换。详情请参考[分布式弹性训练与推理](https://www.mindspore.cn/tutorials/experts/zh-CN/r2.0.0-alpha/parallel/resilience_train_and_predict.html)。

### API变更

#### 算子

- [STABLE] `mindspore.ops.AdaptiveMaxPool3D` 新增算子原语。
- [STABLE] `mindspore.ops.AdjustHue` 新增算子原语。
- [STABLE] `mindspore.ops.BartlettWindow` 新增算子原语。
- [STABLE] `mindspore.ops.BesselJ0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselJ1` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK0e` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK1` 新增算子原语。
- [STABLE] `mindspore.ops.BesselK1e` 新增算子原语。
- [STABLE] `mindspore.ops.BesselY0` 新增算子原语。
- [STABLE] `mindspore.ops.BesselY1` 新增算子原语。
- [STABLE] `mindspore.ops.Betainc` 新增算子原语。
- [STABLE] `mindspore.ops.Bincount` 新增算子原语。
- [STABLE] `mindspore.ops.BlackmanWindow` 新增算子原语。
- [STABLE] `mindspore.ops.Bucketize` 新增算子原语。
- [STABLE] `mindspore.ops.CombinedNonMaxSuppression` 新增算子原语。
- [STABLE] `mindspore.ops.CompareAndBitpack` 新增算子原语。
- [STABLE] `mindspore.ops.Complex` 新增算子原语。
- [STABLE] `mindspore.ops.DataFormatVecPermute` 新增算子原语。
- [STABLE] `mindspore.ops.Eig` 新增算子原语。
- [STABLE] `mindspore.ops.EuclideanNorm` 新增算子原语。
- [STABLE] `mindspore.ops.Expand` 新增算子原语。
- [STABLE] `mindspore.ops.ExtractGlimpse` 新增算子原语。
- [STABLE] `mindspore.ops.FillDiagonal` 新增算子原语。
- [STABLE] `mindspore.ops.FractionalAvgPool` 新增算子原语。
- [STABLE] `mindspore.ops.FractionalMaxPool` 新增算子原语。
- [STABLE] `mindspore.ops.Gcd` 新增算子原语。
- [STABLE] `mindspore.ops.HammingWindow` 新增算子原语。
- [STABLE] `mindspore.ops.Histogram` 新增算子原语。
- [STABLE] `mindspore.ops.HSVToRGB` 新增算子原语。
- [STABLE] `mindspore.ops.Lcm` 新增算子原语。
- [STABLE] `mindspore.ops.LeftShift` 新增算子原语。
- [STABLE] `mindspore.ops.ListDiff` 新增算子原语。
- [STABLE] `mindspore.ops.LogSpace` 新增算子原语。
- [STABLE] `mindspore.ops.Lstsq` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixDiagPartV3` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixDiagV3` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixExp` 新增算子原语。
- [STABLE] `mindspore.ops.MatrixPower` 新增算子原语。
- [STABLE] `mindspore.ops.MaxPool3DWithArgmax` 新增算子原语。
- [STABLE] `mindspore.ops.MaxUnpool2D` 新增算子原语。
- [STABLE] `mindspore.ops.MultilabelMarginLoss` 新增算子原语。
- [STABLE] `mindspore.ops.NextAfter` 新增算子原语。
- [STABLE] `mindspore.ops.Orgqr` 新增算子原语。
- [STABLE] `mindspore.ops.ReduceStd` 新增算子原语。
- [STABLE] `mindspore.ops.ResizeNearestNeighborV2` 新增算子原语。
- [STABLE] `mindspore.ops.RGBToHSV` 新增算子原语。
- [STABLE] `mindspore.ops.RightShift` 新增算子原语。
- [STABLE] `mindspore.ops.Roll` 新增算子原语。
- [STABLE] `mindspore.ops.SampleDistortedBoundingBoxV2` 新增算子原语。
- [STABLE] `mindspore.ops.ScaleAndTranslate` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterAddWithAxis` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdDiv` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMax` 新增算子原语。
- [STABLE] `mindspore.ops.ScatterNdMul` 新增算子原语。
- [STABLE] `mindspore.ops.STFT` 新增算子原语。
- [STABLE] `mindspore.ops.Trace` 新增算子原语。
- [STABLE] `mindspore.ops.UpsampleNearest3D` 新增算子原语。
- [STABLE] `mindspore.ops.UpsampleTrilinear3D` 新增算子原语。
- [STABLE] `mindspore.parallel.transform_checkpoints` 新增分布式权重转换接口。
- [STABLE] `mindspore.parallel.transform_checkpoint_by_rank` 新增分布式权重转换接口。

#### 非兼容性变更

##### Python API

- `mindspore.ms_function`接口名替换为`mindspore.jit`，`mindspore.ms_function` 将在未来版本中弃用并删除。
- `mindspore.ms_class`接口名替换为`mindspore.jit_class`，`mindspore.ms_class` 将在未来版本中弃用并删除。
- `mindspore.ops.ms_kernel`接口名替换为`mindspore.ops.kernel`，`mindspore.ops.ms_kernel` 将在未来版本中弃用并删除。
- `mindspore.dataset.map`接口参数 `column_order` 不再生效，使用`mindspore.dataset.project`替换。
- `mindspore.dataset.close_pool`、`mindspore.dataset.to_device`、`mindspore.dataset.set_dynamic_columns` 接口在之前版本已废弃，当前版本正式删除。

### Bug fixes

- 修复混合精度函数式接口在图模式下不能修改后端驱动的问题。
- 修复以下网络在单P场景下用户可自动传入device_id（mobilenetv1/fasterrcnn/yolov3/yolov4/yolov5/unet/openpose/simplepose/crnn/gnmtv2/faceattribute/facequality/facedetection） 。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.10.1 Release Notes

### 问题修复

- 修复logsumexp防溢出处理中未考虑指定axis的问题
- 修复proto文件的编译依赖问题
- 修复print算子打印结果不正常的问题
- 修复equal算子越界问题
- 修复函数被@jit修饰后，导致的cell_id解析不正确的问题
- 修复GNN场景数据类型校验错误
- 修复Dataset map多进程退化成线程的问题

### 贡献者

感谢以下人员做出的贡献:

archer2049, caifubi, chenfei_mindspore, gaoshuanglong, Greatpan, guozhijian, huoxinyou, Kxiong, lanzhineng, lijunbin, liubuyu, liuchuting, luochao60, lyqlola, nomindcarry, TuDouNi, xiaotianci, xupan, yangshuo, yefeng, YingtongHu, yuchaojie, zhoufeng, ZPaC, 刘勇琪, 吕昱峰, 王禹程, 于振华.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.10.0 Release Notes

### 主要特性和增强

#### DataSet

- [STABLE]下沉模式超时等待时间调整，默认调整到600s，以解决数据下沉模式时因环境资源竞争、计算量大等因素容易导致GetNext算子等待超时的问题。

### Bug fixes

- 修复AMP中部分Primitive算子无法在图模式下实例化导致接口不可用的问题。
- 修复昇腾平台算力切分场景下LSTM网络中DynamicRNN算子执行失败的问题。
- 修复mobilenet, fasterrcnn, yolo等网络单卡训练脚本DEVICE_ID在启动脚本中写死的问题。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.10.0 Release Notes

### Bug fixes

- 修复Arithmetic类CPU算子动态shape场景下可能的计算精度问题。
- 修复Deconv int8量化算子重量化写入地址错误问题。

## MindSpore 1.9.0 Release Notes

### 主要特性和增强

#### FrontEnd

- [STABLE] 新增面向对象+函数式融合编程范式，提供 `mindspore.amp.LossScaler` 、 `mindspore.amp.DynamicLossScaler` 、 `mindspore.amp.StaticLossScaler` 、 `mindspore.amp.auto_mixed_precision` 、 `mindspore.amp.all_finite` 等融合编程范式下的混合精度接口。

### API变更

#### 算子

- [STABLE] `nn.AdaptiveAvgPool3d` 新增nn接口。
- [STABLE] `ops.adaptive_avg_pool3d` 新增functional接口。
- [STABLE] `ops.addcdiv` 新增functional接口。
- [STABLE] `ops.addcmul` 新增functional接口。
- [STABLE] `ops.approximate_equal` 新增GPU、CPU支持。
- [STABLE] `ops.atanh` 新增GPU支持。
- [STABLE] `ops.bessel_i0` 新增GPU支持。
- [STABLE] `ops.bessel_i0e` 新增Ascend支持。
- [STABLE] `ops.bessel_i1` 新增GPU支持。
- [STABLE] `ops.bessel_i1e` 新增Ascend、GPU支持。
- [STABLE] `ops.bessel_j0` 新增GPU支持。
- [STABLE] `ops.bessel_j1` 新增GPU支持。
- [STABLE] `ops.bessel_k0` 新增GPU支持。
- [STABLE] `ops.bessel_k0e` 新增GPU支持。
- [STABLE] `ops.bessel_k1` 新增GPU支持。
- [STABLE] `ops.bessel_k1e` 新增GPU支持。
- [STABLE] `ops.bessel_y0` 新增GPU支持。
- [STABLE] `ops.bessel_y1` 新增GPU支持。
- [STABLE] `ops.bias_add` 新增functional接口。
- [STABLE] `ops.bitwise_and` 新增GPU支持。
- [STABLE] `ops.bitwise_or` 新增GPU支持。
- [STABLE] `ops.bitwise_xor` 新增GPU支持。
- [STABLE] `ops.grid_sample` 新增Ascend支持。
- [STABLE] `ops.inplace_update` 新增CPU支持。
- [STABLE] `ops.isclose` 新增Ascend、GPU支持。
- [STABLE] `ops.isnan` 新增Ascend支持。
- [STABLE] `ops.lerp` 新增GPU支持。
- [STABLE] `ops.random_poisson` 新增functional接口。
- [STABLE] `ops.reverse_sequence` 新增functional接口。
- [STABLE] `ops.scatter_mul` 新增GPU支持。
- [STABLE] `ops.scatter_nd_max` 新增functional接口。
- [STABLE] `ops.scatter_nd_min` 新增functional接口。
- [STABLE] `ops.SparseToDense` 新增GPU支持。
- [STABLE] `ops.square` 新增functional接口。
- [STABLE] `ops.standard_laplace` 新增GPU支持。
- [STABLE] `ops.std` 新增functional接口。
- [STABLE] `ops.trunc` 新增Ascend、GPU支持。
- [STABLE] `ops.unsorted_segment_sum` 新增functional接口。
- [STABLE] `ops.xdivy` 新增functional接口。
- [STABLE] `ops.xlogy` 新增GPU支持。
- `ops.poisson` 接口废弃使用，对应新接口为 `ops.random_poisson` 。
- `ops.SparseApplyAdagrad` 接口废弃使用，可使用 `ops.SparseApplyAdagradV2` 接口替代。

### Bug fixes

- [BUGFIX] 修改混合精度O2 level的判断逻辑，在原来屏蔽 `BatchNorm1d` 、 `BatchNorm2d` 算子的基础上，添加另外两个屏蔽算子`BatchNorm3d`和`LayerNorm`，这4个算子依然用float32数据类型计算。

- [BUGFIX] Dataset处理字符串类型数据时，若调用`create_dict_iterator`或`create_tuple_iterator`接口时指定了`output_numpy=True`，获取到的数据会是`numpy.bytes_`类型。修复此问题后接口会直接返回`numpy.str_`类型数据，用户无需再对其进行字符串解码操作。同样，在使用自定义数据处理函数时，接收到的数据也将直接是`numpy.str_`类型，与原始数据类型相匹配。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, liyanliu, lizhenyu, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, panfengfeng, panyifeng, Payne, peixu_ren, Pengyongrong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanyuan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore 1.8.1 Release Notes

### API变更

#### 算子

- [STABLE] ops.ApplyAdagradDA 新增GPU、CPU支持。
- [STABLE] ops.ApplyAdagradV2 新增CPU支持。
- [STABLE] ops.ApplyCenteredRmsProp 新增Ascend动态shape支持。
- [STABLE] ops.ApplyFtrl 新增CPU支持。
- [STABLE] ops.ApplyGradientDescent 新增CPU支持。
- [STABLE] ops.ApplyPowerSign 新增CPU支持。
- [STABLE] ops.ApplyProximalAdagrad 新增GPU、CPU支持。
- [STABLE] ops.ApplyRmsProp 新增Ascend动态shape支持。
- [STABLE] ops.max 新增functional接口。
- [STABLE] ops.atan2 新增functional接口。
- [STABLE] ops.cummax 新增GPU支持。
- [STABLE] ops.cummin 新增GPU、CPU支持。
- [STABLE] ops.diag 新增GPU支持。
- [STABLE] ops.expand_dims 新增functional接口。
- [STABLE] ops.gather_elements 新增functional接口。
- [STABLE] ops.grid_sample 新增GPU支持。
- [STABLE] ops.hardswish 新增Ascend支持。
- [BETA] ops.index_fill 新增GPU支持。
- [BETA] ops.inplace_update 新增CPU支持。
- [BETA] nn.InstanceNorm1d 新增GPU支持。
- [BETA] nn.InstanceNorm2d 新增GPU支持。
- [BETA] nn.InstanceNorm3d 新增GPU支持。
- [STABLE] ops.log1p 新增functional接口。
- [STABLE] ops.masked_fill 新增GPU、CPU支持。
- [BETA] ops.matrix_diag_part 新增GPU支持。
- [BETA] ops.matrix_diag 新增GPU支持。
- [BETA] ops.matrix_set_diag 新增GPU支持。
- [STABLE] ops.max_pool3d 新增GPU支持。
- [STABLE] ops.nll_loss 新增functional接口。
- [STABLE] ops.one_hot 新增functional接口。
- [STABLE] ops.pad 新增functional接口。
- [STABLE] ops.random_gamma 新增CPU支持。
- [STABLE] ops.amax 新增functional接口。
- [STABLE] ops.mean 新增functional接口。
- [STABLE] ops.amin 新增functional接口。
- [STABLE] ops.prod 新增functional接口。
- [STABLE] ops.renorm 新增Ascend、GPU、CPU支持。
- [BETA] ops.tensor_scatter_elements 新增Ascend、GPU、CPU支持。
- [STABLE] ops.scatter_max 新增GPU支持。
- [STABLE] ops.scatter_min 新增GPU支持。
- [STABLE] ops.scatter_nd 新增functional接口。
- [STABLE] ops.scatter_nd_max 新增GPU支持。
- [STABLE] ops.scatter_update 新增functional接口。
- [STABLE] ops.binary_cross_entropy_with_logits 新增CPU支持。
- [STABLE] ops.smooth_l1_loss 新增functional接口。
- [STABLE] ops.space_to_batch_nd 新增CPU支持。
- [STABLE] ops.SparseApplyAdagrad 新增GPU、CPU支持。
- [STABLE] ops.sparse_segment_mean 新增GPU、CPU支持。
- [STABLE] ops.squeeze 新增functional接口。
- [STABLE] ops.standard_laplace 新增CPU支持。
- [BETA] nn.ReflectionPad1d 新增Ascend、GPU、CPU支持。
- [BETA] nn.ReflectionPad2d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.SiLU 新增Ascend、GPU、CPU支持。
- [STABLE] ops.transpose 新增functional接口。
- [STABLE] ops.uniform_candidate_sampler 新增CPU支持。
- [STABLE] ops.uniform 新增functional接口。
- [STABLE] ops.unique_with_pad 新增GPU支持。
- [STABLE] ops.unstack 新增functional接口。
- [BETA] ops.interpolate 新增GPU、CPU支持。
- [STABLE] ops.xdivy 新增CPU支持。
- [STABLE] ops.xlogy 新增CPU支持。

## MindSpore 1.8.0 Release Notes

### 主要特性和增强

#### FrontEnd

- [BETA]  提供`mindspore.train.Model.fit` API，增加两种callback方法 `mindspore.train.callback.EarlyStopping` 和 `mindspore.train.callback.ReduceLROnPlateau`。
- [BETA] 自定义算子支持Julia算子。
- [BETA] 自定义算子支持Hybrid DSL算子。
- [STABLE] export()接口支持自定义加密算法导出模型，load()接口支持自定义解密算法导入模型。
- [BETA]   [动静统一] [易用性] 图编译支持常量类型设置可变(1.8版本支持tuple/list/dict)。
- [BETA]   [动静统一] 常量场景下控制流内支持JIT Fallback功能。
- [STABLE] [动静统一] 支持图模式常量场景下Python raise语句。
- [STABLE] [动静统一] 支持图模式常量场景下Python assert语句。
- [STABLE] [动静统一] 支持图模式常量场景下Python print语句。
- [STABLE] [动静统一] 支持图模式str.format()方法。
- [STABLE] [动静统一] 支持图模式用slice方法对list赋值。
- [STABLE] [动静统一] 图模式支持创建和调用自定义类的实例。
- [STABLE] [动静统一] 支持从Cell数组/自定义类数组中获取类的属性。
- [STABLE] [动静统一] 图模式下isinstance支持场景扩展。
- [STABLE] 自定义算子修饰符'ms_hybrid'重名为'ms_kernel'。
- [BETA] 自定义算子Hybrid DSL支持CPU后端。
- [BETA] 自定义算子昇腾后端新增自定义调度原语语法支持。

#### PyNative

- [STABLE] 实现AdamWeightDecay算子，替代原有小算子组合方式。
- [STABLE] 动态图下使用动静结合的方式执行优化器。
- [STABLE] 优化PyNative反向图和ms_function的执行性能。

#### Auto Parallel

- [STABLE] 对接AllToAll单算子模式。在KernelByKernel的执行模式下，支持AllToAll算子调用。
- [STABLE] 整图下沉支持MPI启动。整图下沉的模式下，支持使用MPI的方式启动。
- [STABLE] 模型权重的Seed提供并行接口配置。在用户不通过mindspore.set_seed设置随机数种子时，每个参数初始化的随机数种子为当前分片索引决定。当配置随机数种子之后，相同shape以及相同切分策略的权重，其初始化的结果一致。
- [STABLE] HCCL屏蔽内部全连接/非全连接。允许一次训练过程中同时有全连接AllToAllv和分级AllToAllv。
- [BETA] CPU优化器融合。通过优化器跨参数融合，将多个优化器算子按数据类型融合成，带来性能提升。目前已在CPU AdamWeightDecay优化器上做过验证。用户可以通过网络cell类中的flatten_weights方法启用该功能。

#### Executor

- [STABLE] 开放南向芯片对接接口。
- [STABLE] 使用多Actor融合执行提升运行时的执行性能。
- [STABLE] NopOp算子(eg. Reshape)执行消除。
- [STABLE] Embedding Cache架构切换统一分布式运行时。
- [STABLE] Parameter Server训练切换统一分布式运行时。
- [STABLE] 支持CPU Parameter Server模式训练。

#### DataSet

- [STABLE] 对于数据集对象使用map操作时，同时num_parallel_workers>1并且python_multiprocessing=True时，进行了多进程的机制优化，使得数据通道与子进程一一映射，避免了过多的文件句柄占用，同时close_pool这个接口也被删除。
- [STABLE] 新增一批Vision、Text和Audio类数据增强操作。
- [STABLE] 修复数据集类的flat_map方法未将结果展平的错误。
- [STABLE] 统一数据集增强API的导入路径，提供更简单的使用方法，请参阅[最新的API用法](https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.dataset.vision.html)。

### API变更

#### 算子

- [STABLE] ops.adaptive_avg_pool2d 新增GPU支持。
- [BETA] ops.adaptive_max_pool2d  新增Ascend、GPU、CPU支持。
- [BETA] ops.approximate_equal 新增CPU支持。
- [STABLE] ops.argmin 新增CPU支持。
- [BETA] ops.assign_sub 新增CPU支持。
- [STABLE] ops.bernoulli 新增GPU支持。
- [BETA] ops.bessel_i0 新增CPU支持。
- [BETA] ops.bessel_i0e 新增CPU支持。
- [BETA] ops.bessel_i1 新增CPU支持。
- [BETA] ops.bessel_i1e 新增CPU支持。
- [STABLE] ops.bessel_j0 新增CPU支持。
- [STABLE] ops.bessel_j1 新增CPU支持。
- [STABLE] ops.bessel_k0 新增CPU支持。
- [STABLE] ops.bessel_k0e 新增CPU支持。
- [BETA] ops.bessel_k1 新增CPU支持。
- [BETA] ops.bessel_k1e 新增CPU支持。
- [STABLE] ops.bessel_y0 新增CPU支持。
- [STABLE] ops.bessel_y1 新增CPU支持。
- [STABLE] ops.bitwise_and 新增CPU支持。
- [STABLE] ops.bitwise_or 新增CPU支持。
- [STABLE] ops.bitwise_xor 新增CPU支持。
- [STABLE] ops.broadcast_to 新增functional接口。
- [BETA] ops.ceil 新增GPU、CPU支持。
- [BETA] ops.col2im 新增GPU支持。
- [BETA] ops.concat 新增functional接口。
- [STABLE] ops.cosh 新增GPU支持。
- [STABLE] ops.ctc_greedy_decoder 新增Ascend、CPU支持。
- [BETA] ops.DataFormatDimMap 新增GPU、CPU支持。
- [BETA] ops.dropout2d 新增GPU、CPU支持。
- [BETA] ops.dropout3d 新增CPU支持。
- [BETA] ops.erf 新增CPU支持。
- [BETA] ops.erfc 新增CPU支持。
- [STABLE] ops.expand_dims 新增functional接口。
- [STABLE] ops.fast_gelu 新增GPU、CPU支持。
- [STABLE] ops.flatten Ascend动态shape支持。
- [BETA] ops.ger 新增GPU、CPU支持。
- [STABLE] ops.gumbel_softmax 新增Ascend、GPU、CPU支持。
- [BETA] ops.hardshrink 新增GPU、CPU支持。
- [BETA] ops.index_add 新增CPU支持。
- [BETA] ops.inplace_add 新增CPU支持。
- [BETA] ops.inplace_sub 新增CPU支持。
- [STABLE] ops.intopk 新增CPU支持。
- [STABLE] ops.inv 新增GPU、CPU支持。
- [STABLE] ops.invert 新增GPU、CPU支持。
- [BETA] ops.isclose 新增CPU支持。
- [STABLE] ops.lerp 新增CPU支持。
- [BETA] ops.linspace 新增CPU支持。
- [BETA] ops.log_softmax 新增functional接口。
- [BETA] ops.norm 新增Ascend、GPU、CPU支持。
- [BETA] ops.lrn 新增CPU支持。
- [BETA] ops.masked_select 新增GPU支持。
- [BETA] ops.matrix_band_part 新增GPU、CPU支持。
- [BETA] ops.matrix_solve 新增GPU、CPU支持。
- [BETA] ops.meshgrid 新增CPU支持。
- [STABLE] ops.mish 新增CPU支持。
- [BETA] ops.nonzero  新增GPU支持。
- [STABLE] ops.padding 新增GPU、CPU支持。
- [BETA] ops.pow 新增Ascend动态shape支持。
- [BETA] ops.range 新增functional接口。
- [BETA] ops.round 新增Ascend动态shape支持。
- [STABLE] ops.scatter_add 新增Ascend动态shape支持。
- [STABLE] ops.scatter_div 新增Ascend动态shape支持。
- [BETA] ops.scatter_max 新增GPU支持。
- [BETA] ops.scatter_min 新增GPU支持。
- [BETA] ops.scatter_nd_add 新增CPU支持。
- [STABLE] ops.scatter_nd_div 新增GPU、CPU支持。
- [STABLE] ops.scatter_nd_min 新增GPU、CPU支持。
- [STABLE] ops.scatter_nd_mul 新增GPU、CPU支持。
- [BETA] ops.scatter_nd_sub 新增CPU支持。
- [STABLE] ops.scatter_update 新增Ascend动态shape支持。
- [BETA] ops.select 新增Ascend动态shape支持。
- [BETA] ops.selu 新增GPU、CPU支持。
- [BETA] ops.soft_shrink 新增GPU、CPU支持。
- [BETA] ops.softsign 新增CPU支持。
- [STABLE] ops.tan 新增GPU支持。
- [BETA] ops.tensor_scatter_add 新增Ascend、CPU支持。
- [STABLE] ops.tensor_scatter_div 新增GPU、CPU支持。
- [STABLE] ops.tensor_scatter_mul 新增GPU、CPU支持。
- [BETA] ops.tensor_scatter_sub 新增Ascend、CPU支持。
- [STABLE] nn.AdaptiveAvgPool1d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.AdaptiveMaxPool1d 新增Ascend、GPU、CPU支持。
- [BETA] nn.BiDense 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad1d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad2d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ConstantPad3d 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Hardtanh 新增Ascend、GPU、CPU支持。
- [STABLE] nn.HuberLoss 新增Ascend、GPU、CPU支持。
- [STABLE] nn.RReLU 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Tanhshrink 新增Ascend、GPU、CPU支持。
- [STABLE] nn.Threshold 新增Ascend、GPU、CPU支持。
- [STABLE] nn.ZeroPad2d 新增Ascend、GPU、CPU支持。
- [BETA] ops.unique_consecutive 新增GPU支持。
- [STABLE] ops.unsorted_segment_max 新增CPU支持。
- [STABLE] ops.unsorted_segment_min 新增CPU支持。
- [STABLE] ops.unsorted_segment_prod 新增GPU支持。

#### 非兼容性变更

##### Python API

- 不再支持DVPP模拟算法，删除 `mindspore.dataset.vision.c_transforms.SoftDvppDecodeRandomCropResizeJpeg` 和 `mindspore.dataset.vision.c_transforms.SoftDvppDecodeResizeJpeg` 接口。
- LossMonitor中增加`on_train_epoch_end` 方法，实现在 `mindspore.train.Model.fit` 中使用时，打印epoch级别的metric信息。
- TimeMonitor打印内容变更，打印内容加入"train"或"eval"用于区分训练和推理阶段。
- load_checkpoint 接口的`filter_prefix`：不再支持空字符串("")，匹配规则由强匹配修改为模糊匹配。

#### import优化

mindspore.context、mindspore.parallel、mindspore.profiler、mindspore.train模块的接口可直接在mindspore模块使用。原有用法仍可以继续支持。

例如：

- `mindspore.context.set_context`可简化为`mindspore.set_context`。
- `mindspore.parallel.set_algo_parameters`可简化为`mindspore.set_algo_parameters`。
- `mindspore.profiler.Profiler`可简化为`mindspore.Profiler`。
- `mindspore.train.callback.Callback`可简化为`mindspore.train.Callback`。

API页面统一汇总至：<https://www.mindspore.cn/docs/zh-CN/r1.8/api_python/mindspore.html>。

### 贡献者

感谢以下人员做出的贡献：

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking, shu-kun-zhang.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.8.0 Release Notes

### 主要特性和增强

#### API

- [STABLE] 新增模型转换的C++和Python API.
- [STABLE] 新增模型推理的Python API.

#### 后量化

- [STABLE] 后量化支持PerLayer量化，同时内置CLE算法优化精度。

## MindSpore 1.7.0 Release Notes

### 主要特性和增强

#### OS

- [STABLE] 支持Python 3.8版本（Linux/Windows/Mac）。
- [STABLE] 简化安装，提供详细安装指南和自动化安装脚本。
- [STABLE] Windows版本支持算子多线程。
- [STABLE] GCC兼容7.3到9.x版本。

#### FrontEnd

- [STABLE] 优化器支持动态权重衰减，即训练期间权重衰减值随着step的增加而变化。
- [STABLE] 增加四种创建Tensor的方法，分别是`mindspore.numpy.rand()`、`mindspore.numpy.randn()`、`mindspore.numpy.randint()`和`mindspore.ops.arange ()`。
- [STABLE] 增加一种callback方法 `mindspore.train.callback.History`。
- [BETA] 自定义算子支持Julia算子。
- [STABLE] 通过 `mindspore.ms_class` 类装饰器，支持获取用户自定义类的属性和方法。
- [STABLE] 支持同时存在副作用算子和控制流语句的网络的训练。
- [STABLE] 支持更复杂的控制流语法，比如在while的循环体里使用for语句。
- [STABLE] 通过减少子图数量，提升包含复杂控制流语法的网络的性能。

#### PyNative

- [STABLE] 在PyNative模式下支持hook函数功能，包括前向hook接口register_forward_pre_hook、register_forward_hook和反向hook接口register_backward_hook。
- [STABLE] 优化PyNative模式执行性能，并行执行前端Python与后端C++。

#### Auto Parallel

- [STABLE] 在MoE场景中支持TopK的路由、数据并行和优化器切分。
- [STABLE] 支持AllGather/ReduceScatter通信算子融合，在DATA_PARALLEL模式支持AllReduce按数据量大小编译。
- [STABLE] 在并行模式下支持ops.clip_by_global_norm。
- [STABLE] 在并行模式下支持AdaSum优化器。
- [STABLE] 支持自动优化器切分。
- [STABLE] 支持AlltoAll可配置开启，支持自动插入VirtualDatasetCell。
- [STABLE] 在流水线并行训练中，支持自动推断可训练的参数。
- [STABLE] 支持集群的设备数目不为2的幂次方。
- [STABLE] 在自动并行模式中支持策略传播。
- [STABLE] 在统一运行时中支持异构训练。
- [STABLE] 支持CPU的Adafactor算子。
- [STABLE] 支持Conv2d/Conv2D的H/W轴切分和Transpose算子。支持ResizeBilinear、ROIAlign、CropAndResize、BoundingBoxEncode、IOU和RandomChoiceWithMask等分布式算子。

#### Executor

- [BETA] [数据并行训练容灾](https://www.mindspore.cn/tutorials/experts/zh-CN/r1.7/parallel/train_gpu.html#%E5%AE%B9%E7%81%BE%E6%81%A2%E5%A4%8D) 支持多卡数据并行训练容灾恢复。
- [BETA] 支持在CPU下的线程数搜索，获取最优线程数来执行。整个搜索过程需要耗时50个steps，整体的性能会在50个steps后达到稳定的状态。在测试性能的时候，需要以50个steps之后的数据作为标准。

#### DataSet

- [STABLE] 增加了数据处理API的差异文档，比较TensorFlow.data与MindSpore.dataset部分算子的差异，详见 [对比文档](https://www.mindspore.cn/docs/zh-CN/r1.7/note/api_mapping/tensorflow_api_mapping.html#tf-data)。
- [STABLE] Python多进程逻辑优化，保证不同异常场景的正常退出。
- [STABLE] 支持[自动数据加速](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/dataset_autotune.html)，可以自适应调节数据处理管道的执行速度。
- [BETA] [数据处理异构加速](https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/dataset_offload.html) 支持了新的数据增强操作: RandomColorAdjust、RandomSharpness和TypeCast。
- GeneratorDataset加载自定义数据集时，当`__getitem__/__next__`方法返回单个NumPy对象，对应会输出单个数据列。
- 用户在数据预处理中使用过多的进程数/线程数情况下，会出现错误RuntimeError: can't start new thread，可以通过 `ulimit -u 10240` 增加当前用户可用的线程/进程数解决。

### API变更

#### 非兼容性变更

##### Python API

- 修改register_backward_hook功能对应hook的梯度返回值类型，将梯度返回值统一改成tuple类型。([!31876](https://gitee.com/mindspore/mindspore/pulls/31876))
- 弃用的import用法： `import mindspore.dataset.engine.datasets as ds` ，因其import目录过深且过度依赖Python目录结构。推荐使用 `import mindspore.dataset as ds` ，更多参考详见 [API文档](https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/mindspore.dataset.html)。
- 新增`mindspore.ms_class` 接口，作为用户自定义类的类装饰器，使得MindSpore能够识别用户自定义类，并且获取这些类的属性和方法。([!30855](https://gitee.com/mindspore/mindspore/pulls/30855))
- `mindspore.SparseTensor`接口废弃使用，对应新接口为`mindspore.COOTensor`。 ([!28505](https://gitee.com/mindspore/mindspore/pulls/28505))
- Tensor新增一个入参`internal`，作为框架内部使用。

### 贡献者

感谢以下人员做出的贡献:

AGroupofProbiotocs, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jiabin Liu, jianghui58, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, liuyongqi, laiyongqiang, leonwanghui, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, lvchangquan, lvliang, lz, maning202007, Margaret_wangrui, mengyuanli, Ming_blue, ms_yan, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin,  wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiao Tianci, Xiaoda, xiefangqi, xinyunfan, xuanyue, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang,  zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Ziyan, zjun, ZPaC, wangfengwfwf, zymaa, gerayking.

欢迎以任何形式对项目提供贡献！

## MindSpore Lite 1.7.0 Release Notes

### 主要特性和增强

#### 后量化

- [STABLE] 后量化支持动态量化算法。
- [BETA] 后量化模型支持在英伟达GPU上执行推理。
