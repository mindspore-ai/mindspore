# MindSpore 1.2.0

## MindSpore 1.2.0 Release Notes

### Major Features and Improvements

#### NewModels

- [STABLE] Add CV models on Ascend: 3D Unet, Unet++, SSD-Resnet50-fpn, SSD-VGG16, crnn_seq2seq_ocr for BSI, CTPN, resnet18, DPN
- [STABLE] Add CV models on GPU: Faster-RCNN
- [STABLE] Add NLP models on Ascend: NAML, Fasttext, GRU, LSTM
- [BETA] Add TPRR: Thinking Path Re-Ranker, an original ranked-base framework for Multi-Hop Question Answering which has won the first place in HotpotQA leaderboard.(Ascend)

#### FrontEnd

- [STABLE] Support side effects expression to ensure that the perform order of user's semantics is correct.(Ascend/GPU/CPU)
- [STABLE] Support calculating the gradient for network that contain non-Tensor input parameters（int, float, bool, mstype,int, mstype.float, mstype.uint, mstype.bool_, tuple, list, dict）.(Ascend/GPU/CPU)
- [STABLE] Support the inverse of a bool Tensor.(Ascend/GPU/CPU)
- [STABLE] Uniform the interface `isinstance`.(Ascend/GPU/CPU)
- [STABLE] Support negative indexes.(Ascend/GPU/CPU)
- [STABLE] Support 110+ Numpy-like interfaces in mindspore.numpy.(Ascend/GPU/CPU)
- [STABLE] Support export/load mindir model with a size greater than 2 GB.
- [STABLE] The optimizer supports gradient centralization.(Ascend)
- [STABLE] Support support auc metric, rou metric, bleu score metric, confusion matrix metric, cosine similarity metric, dice metric, hausdorff distance metric, occlusion sensitivity metric, perplexity metric, mean surface distance metric, root mean surface distance metric.
- [STABLE] Support use EmbeddingLookup with cache.(Ascend)

#### Auto Parallel

- [STABLE] Support AllGather and ReduceScatter fusion.(Ascend)
- [STABLE] Support gradient accumulation feature in auto parallel mode.(Ascend/GPU)
- [STABLE] Support running parallel optimizer with gradient accumulation.(Ascend)
- [STABLE] Add the configuration of communication operators' fusion.(Ascend)

#### Executor

- [STABLE] Support inference with Nvidia GPU.
- [STABLE] Support data parallelism in PyNative mode.(Ascend/GPU)
- [STABLE] Optimize LSTM inference memory consumption in Graph mode with CPU.

#### Sponge

- [STABLE] Add SPONGE modules for molecular dynamics simulation, including Bond, Angle, Dihedral, Non Bond 14, NeighborList, Particle Mesh Ewald, Langevin MD and LIUJIAN MD.(GPU)

#### DataSet

- [STABLE] If the libnuma library is installed in the environment, you can run `export DATASET_ENABLE_NUMA=True` to configure NUMA binding. In multi-card training scenarios, the training data processing speed can be improved, thereby improving the network training efficiency.
- [STABLE] Unify API Tensor structure of Training/Inference interfaces in C++ SDK.
- [STABLE] Optimize duplicated Decode in data preprocess using cache, improve preprocess efficiency.
- [STABLE] Support eager mode to run data augmentation in Python & C++.
- [STABLE] Support more data augmentation operators(e.g. Affine, Perspective) in MindSpore-Lite.
- [STABLE] Support light pipeline to process MindData in MindSpore-Lite training.
- [STABLE] Support more data preprossing operators based on DVPP hardware module and can be used on on Ascend310 platform.
- [STABLE] Support copy-free property for data in Ascend310 inference process scenarios.

#### Running Data Recorder

- [STABLE] Support running data recorder (RDR)  for exception demarcation.
- [STABLE] Provide records of multi-stage computational graphs, memory allocation information, graph execution order, stream execution order and task debug information when a "run task error" or "distribute task failed" occurs. (Ascend)
- [STABLE] Provide records of multi-stage computational graphs, memory allocation information and graph execution order when a "SyncStream error" occurs. (GPU)

#### 3D Feature

- [STABLE] Support 3D ops: Conv3D, Conv3DBackpropInput, Conv3DBackpropFilter, Conv3DTranspose, BiasAdd, BiasAddGrad, PReLU, Transpose, Reshape, transdata, StrideSlice, MaxPool3D, MaxPool3DGrad, BinaryCrossEntropy, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsGrad, SoftmaxCrossEntropyWithLogits, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsGrad, BatchNorm3d, BatchNorm3dGrad, Dropout3d.
- [STABLE] Support RMSELoss loss function, MAELoss loss function, FocalLoss loss function, DiceLoss binary loss function, and MultiClassDiceLoss multi-type loss function for 2D/3D network.
- [STABLE] Add optimizer: AdamApplyOne(3D), ApplyMomentum(3D), SGD(3D).

### API Change

#### Backwards Incompatible Change

##### Python API

###### `mindspore.numpy.array()`, `mindspore.numpy.asarray()`, `mindspore.numpy.asfarray()`, `mindspore.numpy.copy()` now support GRAPH mode, but cannot accept `numpy.ndarray` as input arguments anymore([!12726](https://gitee.com/mindspore/mindspore/pulls/12726))

Previously, these interfaces can accept numpy.ndarray as arguments and convert numpy.ndarray to Tensor, but cannot be used in GRAPH mode.
However, currently MindSpore Parser cannot parse numpy.ndarray in JIT-graph. To support these interfaces in graph mode, we have to remove `numpy.ndarray` support. With that being said, users can still use `Tensor` to convert `numpy.ndarray` to tensors.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.numpy as mnp
>>> import numpy
>>>
>>> nd_array = numpy.array([1,2,3])
>>> tensor = mnp.asarray(nd_array) # this line cannot be parsed in GRAPH mode
```

</td>
<td>

```python
>>> import mindspore.numpy as mnp
>>> import numpy
>>>
>>> tensor = mnp.asarray([1,2,3]) # this line can be parsed in GRAPH mode
```

</td>
</tr>
</table>

###### mindspore.numpy interfaces remove support for keyword arguments `out` and `where`([!12726](https://gitee.com/mindspore/mindspore/pulls/12726))

Previously, we have incomplete support for keyword arguments `out` and `where` in mindspore.numpy interfaces, however, the `out` argument is only functional when `where` argument is also provided, and `out` cannot be used to pass reference to numpy functions. Therefore, we have removed these two arguments to avoid any confusion users may have. Their original functionality can be found in [np.where](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/numpy/mindspore.numpy.where.html#mindspore.numpy.where)

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.numpy as np
>>>
>>> a = np.ones((3,3))
>>> b = np.ones((3,3))
>>> out = np.zeros((3,3))
>>> where = np.asarray([[True, False, True],[False, False, True],[True, True, True]])
>>> res = np.add(a, b, out=out, where=where) # `out` cannot be used as a reference, therefore it is misleading
```

</td>
<td>

```python
>>> import mindspore.numpy as np
>>>
>>> a = np.ones((3,3))
>>> b = np.ones((3,3))
>>> out = np.zeros((3,3))
>>> where = np.asarray([[True, False, True],[False, False, True],[True, True, True]])
>>> res = np.add(a, b)
>>> out = np.where(where, x=res, y=out) # instead of np.add(a, b, out=out, where=where)
```

</td>
</tr>
</table>

###### Turn `ops.MakeRefKey` into an internal interface ([!12010](https://gitee.com/mindspore/mindspore/pulls/12010))

Previously MakeRefKey is an external interface that is not used, now make it an internal interface with the same usage. We do not recommend users to use this interface, and we will remove the relevant introduction of this interface from the official website.

###### `ops.ApplyFtrl`, `ops.ApplyMomentum`, `ops.ApplyRMSProp`, `ops.ApplyCenteredRMSProp` change the output on Ascend backend from multiple to a single. ([!11895](https://gitee.com/mindspore/mindspore/pulls/11895))

Previously the number of outputs of these operator is different on different backends. To unify their definition we change their output on Ascend backend from multiple to a single.

##### `P.FusedBatchNorm`, `P.FusedBatchNormEx` deleted ([!12115](https://gitee.com/mindspore/mindspore/pulls/12115))

The FusedBatchNorm and FusedBatchNormEx interface has been deleted. Please use the batchnorm operator to replace it.

##### `MetaTensor` deleted ([!10325](https://gitee.com/mindspore/mindspore/pulls/10325))

The MetaTensor interface has been deleted. The function of MetaTensor has been integrated into tensor.

###### `ControlDepend` is deleted, use `Depend` instead. The decorator `@C.add_flags(has_effect=True)` does not work. ([!13793](https://gitee.com/mindspore/mindspore/pulls/13793))

Previously, we used ControlDepend to control the execution order of multiple operators. In version 1.2.0, mindspore introduces the auto-monad side effects expression to ensure that the perform order of user's semantics is correct. Therefore, ControlDepend is deleted and Depend is recommended.

In most scenarios, if operators have IO side effects (such as print) or memory side effects (such as assign), they will be executed according to the user's semantics. In some scenarios, if the two operators A and B have no order dependency, and A must be executed before B, we recommend using Depend to specify their execution order. See the API documentation of the Depend operator for specific usage.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
    In some side-effect scenarios, we need to ensure the execution order of operators.
    In order to ensure that operator A is executed before operator B, it is recommended
    to insert the Depend operator between operators A and B.

    Previously, the ControlDepend operator was used to control the execution order.
    Since the ControlDepend operator is deprecated from version 1.1, it is recommended
    to use the Depend operator instead. The replacement method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
        ControlDepend(a, b)     --->        b = B(y)
```

</td>
<td>

```python
    In most scenarios, if operators have IO side effects or memory side effects,
    they will be executed according to the user's semantics. In some scenarios,
    if the two operators A and B have no order dependency, and A must be executed
    before B, we recommend using Depend to specify their execution order. The
    usage method is as follows::

        a = A(x)                --->        a = A(x)
        b = B(y)                --->        y = Depend(y, a)
                                --->        b = B(y)
```

</td>
</tr>
</table>

After the introduction of the auto-monad side effect expression feature, the decorator `@C.add_flags(has_effect=True)` does not work. If the decorator is used in the script, please modify. Take the overflow identification operator (without side effects) as an example, the modification method is as follows:

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
@C.add_flags(has_effect=True)
def construct(self, *inputs):
    ...
    loss = self.network(*inputs)
    init = self.allo_status()
    self.clear_status(init)
    ...
```

</td>
<td>

```python
def construct(self, *inputs):
    ...
    loss = self.network(*inputs)
    init = self.allo_status()
    init = F.depend(init, loss)
    clear_status = self.clear_status(init)
    ...
```

</td>
</tr>
</table>

##### C++ API

###### C++ API support dual ABI now.([!12432](https://gitee.com/mindspore/mindspore/pulls/12432))

1.1.1 supports only the old ABI. Currently, both the new and the old are supported.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
```

</td>
<td>

```cmake
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)  # old ABI are supported
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)  # new ABI are supprrted, too
                                                   # write nothing, use new ABI as default
```

</td>
</tr>
</table>

###### Context refactor.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

The `Context` class is refactored. For details, see the API docs.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
GlobalContext::SetGlobalDeviceTarget(kDeviceTypeAscend310);       // set device target is ascend310
GlobalContext::SetGlobalDeviceID(0);                              // set device id is 0
auto model_context = std::make_shared<ModelContext>();            // create a model context
ModelContext::SetInsertOpConfigPath(model_context, "./aipp.cfg")  // set aipp config file is ./aipp.cfg
```

</td>
<td>

```cpp
auto model_context = std::make_shared<Context>();                 // create a model context
auto ascend310_info = std::make_shared<Ascend310DeviceInfo>();
model_context.MutableDeviceInfo().push_back(ascend310_info );     // set device target is ascend310
ascend310_info->SetDeviceID(0);                                   // set device id is 0
ascend310_info->SetInsertOpConfigPath("./aipp.cfg");              // set aipp config file is ./aipp.cfg
```

</td>
</tr>
</table>

###### LoadModel interface changes.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`LoadModel` is renamed `Load`. No exception is thrown new but the return status should be checked.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
try {
  auto graph = Serialization::LoadModel(model_file_path, kMindIR);
} catch (...) { ... }
```

</td>
<td>

```cpp
Graph graph;
auto ret = Serialization::Load(model_file_path, kMindIR, &graph);
if (ret != kSuccess) { ... }
```

</td>
</tr>
</table>

###### Model ctor changes.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`Model` uses a non-parameter ctor now, and arguments are passed in through `Build`.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
Model net(net_cell, model_context);
auto ret = net.Build();
if (ret != kSuccess) { ... }
```

</td>
<td>

```cpp
Model net;
auto ret = net.Build(net_cell, model_context);
if (ret != kSuccess) { ... }
```

</td>
</tr>
</table>

###### MSTensor::CreateTensor returns a native pointer now.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

`MSTensor::CreateTensor` and `MSTensor::CreateRefTensor` returns a native pointer now, need to be destroy by `DestroyTensorPtr`.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
auto tensor = MSTensor::CreateTensor(xxx, xxx, ...);
auto name = tensor.Name();
```

</td>
<td>

```cpp
auto tensor = MSTensor::CreateTensor(xxx, xxx, ...);
auto name = tensor->Name();
MSTensor::DestroyTensorPtr(tensor);
```

</td>
</tr>
</table>

#### New features

##### Python API

- Add SPONGE functions: `mindspore.ops.operations.BondForceWithAtomEnergy`, `mindspore.ops.operations.AngleForceWithAtomEnergy`, `mindspore.ops.operations.DihedralForceWithAtomEnergy`, `mindspore.ops.operations.Dihedral14LJCFForceWithAtomEnergy`, `mindspore.ops.operations.LJForceWithPMEDirectForce`, `mindspore.ops.operations.PMEExcludedForce`, `mindspore.ops.operations.PMEReciprocalForce`,`mindspore.ops.operations.BondEnergy`, `mindspore.ops.operations.AngleEnergy`,`mindspore.ops.operations.DihedralEnergy`, `mindspore.ops.operations.Dihedral14LJEnergy`, `mindspore.ops.operations.Dihedral14CFEnergy`,`mindspore.ops.operations.LJEnergy`, `mindspore.ops.operations.PMEEnergy`. All operators are supported in `GPU`.

#### Deprecations

##### Python API

###### `nn.MatMul` is now deprecated in favor of `ops.matmul` ([!12817](https://gitee.com/mindspore/mindspore/pulls/12817))

[ops.matmul](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/ops/mindspore.ops.matmul.html#mindspore.ops.matmul) follows the API of [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html) as closely as possible. As a function interface, [ops.matmul](https://www.mindspore.cn/doc/api_python/zh-CN/master/mindspore/ops/mindspore.ops.matmul.html#mindspore.ops.matmul) is applied without instantiation, as opposed to `nn.MatMul`, which should only be used as a class instance.

<table>
<tr>
<td style="text-align:center"> 1.1.1 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import numpy as np
>>> from mindspore import Tensor, nn
>>>
>>> x = Tensor(np.ones((2, 3)).astype(onp.float32)
>>> y = Tensor(np.ones((3, 4)).astype(onp.float32)
>>> nn.MatMul()(x, y)
```

</td>
<td>

```python
>>> import numpy as np
>>> from mindspore import Tensor, ops
>>>
>>> x = Tensor(np.ones((2, 3)).astype(onp.float32)
>>> y = Tensor(np.ones((3, 4)).astype(onp.float32)
>>> ops.matmul(x, y)
```

</td>
</tr>
</table>

### Bug fixes

#### FrontEnd

- fix the null pointer problem of evaluator in control flow.([!13312](https://gitee.com/mindspore/mindspore/pulls/13312))
- fix parameter naming conflict bug for CellList and SequentialCell. ([!13260](https://gitee.com/mindspore/mindspore/pulls/13260))

#### Executor

- fix executor pending task not execute in some heterogeneous cases.([!13465](https://gitee.com/mindspore/mindspore/pulls/13465))
- add passes to support frontend IR unification, including following operations: SliceGrad([!11783](https://gitee.com/mindspore/mindspore/pulls/11783)), ApplyFtrl, ApplyMomentum, ApplyRMSProp, CenteredRMSProp([!11895](https://gitee.com/mindspore/mindspore/pulls/11895)), AvgPoolGrad([!12813](https://gitee.com/mindspore/mindspore/pulls/12813)), BatchNorm([!12115](https://gitee.com/mindspore/mindspore/pulls/12115))

#### Dataset

- Fix getter functions(e.g. GetDatasetSize) terminated abnormally when use python multi-processing. ([!13571](https://gitee.com/mindspore/mindspore/pulls/13571), [!13823](https://gitee.com/mindspore/mindspore/pulls/13823))
- Fix unclear error log of data augmentation operators. ([!12398](https://gitee.com/mindspore/mindspore/pulls/12398), [!12883](https://gitee.com/mindspore/mindspore/pulls/12883), [!13176](https://gitee.com/mindspore/mindspore/pulls/13176))
- Fix profiling performs abnormally when sink_size = False, as saving data is later than profiling analysis. ([!13944](https://gitee.com/mindspore/mindspore/pulls/13944))

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Support TensorFlow model in Converter except aware-training model.
2. Add fusion pattern for same horizontal operators in Converter.
3. Support Jar in x86_64 system for integrating into server with Java backend conveniently.
4. Provide unified runtime API for developer reusing their code between cloud side and end side.[BETA]
5. Improve control-flow capabilities continually: Support GRU fusion in Converter; Support weight-quant for control-flow model; Support control-flow model inference with half precision; Support nested control-flow model.[BETA]

#### ARM backend optimization

1. Add NLP dependent float16 operators(like lstm) to enhance inference performance.
2. Optimize operators: lstm, gru, depthwise.
3. Add 6 NPU operators(like FullConnection), and fix some bugs about buildIR failed.

#### OpenCL backend

1. Add new ops：add 10+ ops，total 72 ops；
2. Performance optimization：by memory layout optimize，block tiling，Performance improved by 30% compared to version 1.1 at Adreno GPU.
3. Initialization time optimization：initialization time improve 100% vs MSLITE Version1.1 by store kernel cache as binary.
4. Support Java call on Mali or Adreno GPU.

#### Post quantization

1. Support quantization of gather and lstm ops.
2. Support quantizatizing TF Lite models with sub-graph node.
3. Add quantiztion strategy to decide quantize ops or not，less accuracy loss and higher compression rate.

#### Training on Device

1. Virtual batching, use mini-batch to minic large batch in theorical with few RAM consumption.
2. Converter unify, do not compile tod and iod converter separately.
3. Performance optimization to BWD ops.
4. TrainLoop with Off-The-Shelf Functionality blocks, like LR scheduler, Loss Monitor, Ckpt Saver, Accuracy Monitor.
5. Integration of code with Minddata lite.
6. Support more networks (googlenet, densenet, shufflenetv2, nin, vgg) and operators.

#### Codegen

1. Support 79 ops for the ARM platform and all CMSIS ops for Arm Cortex-M Series.
2. Multiplatform support, including Android, IoT Devices.
3. Support offline model weight preprocessing while compiling.
4. Support offline memory reuse computing for minimum runtime buffer size.

### API Change

#### API Incompatible Change

##### C++ API

###### Add header file named lite_types.h for some common data structs. ([!12262](https://gitee.com/mindspore/mindspore/pulls/12262))

Previously, some common data structs such as `CpuBindMode` and `DeviceType` are in context.h, this may cause cross-dependency between headers. So we create a new header named lite_types.h for some common data structs and move `CpuBindMode` and `DeviceType` from context.h into lite_types.h.

<table>
<tr>
<td style="text-align:center"> lite_types.h </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief CpuBindMode defined for holding bind cpu strategy argument.
typedef enum {
  NO_BIND,    /**< no bind */
  HIGHER_CPU, /**< bind higher cpu first */
  MID_CPU     /**< bind middle cpu first */
} CpuBindMode;

/// \brief DeviceType defined for holding user's preferred backend.
typedef enum {
  DT_CPU, /**< CPU device type */
  DT_GPU, /**< GPU device type */
  DT_NPU  /**< NPU device type */
} DeviceType;
}  // namespace mindspore::lite
```

</td>
</tr>
</table>

###### Add some new interfaces in ms_tensor.h for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, users could not create `MSTensor` or modify ``MSTensor, all `MSTensor` are created and managed by framework. However users need to create or modify MSTensor sometimes such as pre-processing input data. So we provide two new interfaces in ms_tensor.h: `CreateTensor` interface for creating `MSTensor` by user and `set_shape` interface for modifying the shape of `MSTensor`.

<table>
<tr>
<td style="text-align:center"> CreateTensor </td>
</tr>
<tr>
<td>

```cpp
/// \brief Create a MSTensor.
///
/// \return Pointer to an instance of MindSpore Lite MSTensor.
static MSTensor *CreateTensor(const std::string &name, TypeId type, const std::vector<int> &shape, const void *data,
                                size_t data_len);
```

</td>
</tr>
</table>

<table>
<tr>
<td style="text-align:center"> set_shape </td>
</tr>
<tr>
<td>

```cpp
/// \brief Set the shape of MSTensor.
virtual void set_shape(const std::vector<int> &shape) = 0;
```

</td>
</tr>
</table>

Previously, users could access to data of `MSTensor` by interface named `MutableData`. However `MutableData` is not only returning data of tensor but also allocating data for tensor if its data is nullptr. So we provide a new interfaces in ms_tensor.h named `data` for returning data of tensor without allocating automatically.

<table>
<tr>
<td style="text-align:center"> data </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get the pointer of data in MSTensor.
///
/// \note The data pointer can be used to both write and read data in MSTensor. No memory buffer will be
/// allocated.
///
/// \return the pointer points to data in MSTensor.
virtual void *data() = 0;
```

</td>
</tr>
</table>

###### Delete `DimensionSize()` in ms_tensor.h.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

The interface named `DimensionSize` is fuinctionally overlapped with the interface named `shape`. For the simplicity of the interface, we delete `DimensionSize` and recommend users to use the new interface named `shape` instead.

<table>
<tr>
<td style="text-align:center"> DimensionSize() </td>
</tr>
<tr>
<td>

```cpp
/// \brief Get size of the dimension of the MindSpore Lite MSTensor index by the parameter index.
///
/// \param[in] index Define index of dimension returned.
///
/// \return Size of dimension of the MindSpore Lite MSTensor.
virtual int DimensionSize(size_t index) const = 0;
```

</td>
</tr>
</table>

###### Move allocator from namespace mindspore::lite to namespace lite for unified runtime API.([!13515](https://gitee.com/mindspore/mindspore/pulls/13515))

Previously, class `Allocator` is in namespace mindspore::lite. Considering unified allocator interface for unified runtime API, we move `Allocator` to namespace mindspore.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.2.0 </td>
</tr>
<tr>
<td>

```cpp
namespace mindspore::lite {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
<td>

```cpp
namespace mindspore {
/// \brief Allocator defined a memory pool for malloc memory and free memory dynamically.
///
/// \note List public class and interface for reference.
class Allocator;
}
```

</td>
</tr>
</table>

### Bug fixes

1. Fix the bug that the array in kernel registrar is not initialized.
2. Fix segment fault caused by releasing of OpParameter in Crop kernel in mistake.
3. Fix the bug that the MINDIR aware-training model is finally interpreted as weight-quant model.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, dong-li001, eric, Eric, fary86, fuzhiye, Gaoxiong, GAO_HYP_XYJ, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Islam Amin, Jesse, , Jiabin Liu, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, Lin Xh, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luopengting, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, Ming_blue, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, qianjiahong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wudenggang, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhanghui_china, zhangxinfeng3, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhiqwang, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa.

Contributions of any kind are welcome!

# MindSpore 1.1.1 Release Notes

## MindSpore

### Major Features and Improvements

#### NewModels

- [STABLE] BGCF: a Bayesian Graph Collaborative Filtering(BGCF) framework used to model the uncertainty in the user-item interaction graph and thus recommend accurate and diverse items on Amazon recommendation dataset.(Ascend)
- [STABLE] GRU: a recurrent neural network architecture like the LSTM(Long-Short Term Memory) on Multi30K dataset.(Ascend)
- [STABLE] FastText: a simple and efficient text classification algorithm on AG's news topic classification dataset, DBPedia Ontology classification dataset and Yelp Review Polarity dataset.(Ascend)
- [STABLE] LSTM: a recurrent neural network architecture used to learn word vectors for sentiment analysis on aclImdb_v1 dataset.(Ascend)
- [STABLE] SimplePoseNet: a convolution-based neural network for the task of human pose estimation and tracking on COCO2017 dataset.(Ascend)

#### FrontEnd

- [BETA] Support Tensor Fancy Index Getitem with tuple and list. (Ascend/GPU/CPU)

### Backwards Incompatible Change

#### Python API

##### `ops.AvgPool`, `ops.MaxPool`, `ops.MaxPoolWithArgmax` change attr name from 'ksize', 'padding' to 'kernel_size', 'pad_mode' ([!11350](https://gitee.com/mindspore/mindspore/pulls/11350))

Previously the kernel size and pad mode attrs of pooling ops are named "ksize" and "padding", which is a little puzzling and inconsistent with convolution ops. So they are rename to "kernel_size" and "pad_mode".

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(ksize=2, padding='same')
>>> max_pool = ops.MaxPool(ksize=2, padding='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(ksize=2, padding='same')
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> avg_pool = ops.AvgPool(kernel_size=2, pad_mode='same')
>>> max_pool = ops.MaxPool(kernel_size=2, pad_mode='same')
>>> max_pool_with_argmax = ops.MaxPoolWithArgmax(kernel_size=2, pad_mode='same')
```

</td>
</tr>
</table>

##### `ops.TensorAdd`, change API name to `ops.Add` ([!11568](https://gitee.com/mindspore/mindspore/pulls/11568))

The operator name TensorAdd is not standardized, it is changed to Add. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.TensorAdd()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> add = ops.Add()
```

</td>
</tr>
</table>

##### `ops.Gelu`, `ops.GeluGrad`, `ops.FastGelu`, `ops.FastGeluGrad`, change API name to `ops.GeLU`, `ops.GeLUGrad`, `ops.FastGeLU`, `ops.FastGeLUGrad` ([!11603](https://gitee.com/mindspore/mindspore/pulls/11603))

Gelu, GeluGrad, FastGelu, and FastGeluGrad names are unified into ReLU naming rules, "lu" is changed to the uppercase "LU". The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.Gelu()
>>> gelu_grad = ops.GeluGrad()
>>> fast_gelu = ops.FastGelu()
>>> fast_gelu_grad = ops.FastGeluGrad()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gelu = ops.GeLU()
>>> gelu_grad = ops.GeLUGrad()
>>> fast_gelu = ops.FastGeLU()
>>> fast_gelu_grad = ops.FastGeLUGrad()
```

</td>
</tr>
</table>

##### `ops.GatherV2`, change API name to `ops.Gather` ([!11713](https://gitee.com/mindspore/mindspore/pulls/11713))

GatherV2 is changed to Gather. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.GatherV2()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> gather = ops.Gather()
```

</td>
</tr>
</table>

##### `ops.Pack`、`ops.Unpack`, change API name to `ops.Stack`、`ops.Unstack` ([!11828](https://gitee.com/mindspore/mindspore/pulls/11828))

Pack is changed to Stack, and Unpack is changed to Unstack. The old interface can be used continuously, but will be deleted in subsequent versions, it is recommended to use and switch to the latest interface.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> pack= ops.Pack()
>>> unpack= ops.Unpack()
```

</td>
<td>

```python
>>> import mindspore.ops as ops
>>>
>>> stack= ops.Stack()
>>> unstack= ops.Unstack()
```

</td>
</tr>
</table>

##### `ops.ControlDepend`, add deprecated to ControlDepend ([!11844](https://gitee.com/mindspore/mindspore/pulls/11844))

ControlDepend is deprecated and will be removed in a future version, use Depend instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```pythonNote:
Note:
    This operation does not work in `PYNATIVE_MODE`.
```

</td>
<td>

```python
Note:
        This operation does not work in `PYNATIVE_MODE`.
        `ControlDepend` is deprecated from version 1.1 and will be removed in a future version, use `Depend` instead.
```

</td>
</tr>
</table>

##### `ops.Depend`, add operator description and use case ([!11815](https://gitee.com/mindspore/mindspore/pulls/11815)), ([!11879](https://gitee.com/mindspore/mindspore/pulls/11879))

Since the ControlDepend operator will be deprecated from version 1.2, it is recommended to use the Depend operator instead.

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```python
Depend is used for processing side-effect operations.

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``
```

</td>
<td>

```python
Depend is used for processing dependency operations.

In some side-effect scenarios, we need to ensure the execution order of operators.
In order to ensure that operator A is executed before operator B, it is recommended
to insert the Depend operator between operators A and B.

Previously, the ControlDepend operator was used to control the execution order.
Since the ControlDepend operator will be deprecated from version 1.2, it is
recommended to use the Depend operator instead. The replacement method is as follows::

    a = A(x)                --->        a = A(x)
    b = B(y)                --->        y = Depend(y, a)
    ControlDepend(a, b)     --->        b = B(y)

Inputs:
    - **value** (Tensor) - the real value to return for depend operator.
    - **expr** (Expression) - the expression to execute with no outputs.

Outputs:
    Tensor, the value passed by last operator.

Supported Platforms:
    ``Ascend`` ``GPU`` ``CPU``

Examples:
    >>> import numpy as np
    >>> import mindspore
    >>> import mindspore.nn as nn
    >>> import mindspore.ops.operations as P
    >>> from mindspore import Tensor
    >>> class Net(nn.Cell):
    ...     def __init__(self):
    ...         super(Net, self).__init__()
    ...         self.softmax = P.Softmax()
    ...         self.depend = P.Depend()
    ...
    ...     def construct(self, x, y):
    ...         mul = x - y
    ...         y = self.depend(y, mul)
    ...         ret = self.softmax(y)
    ...         return ret
    ...
    >>> x = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> y = Tensor(np.ones([4, 5]), dtype=mindspore.float32)
    >>> net = Net()
    >>> output = net(x, y)
    >>> print(output)
    [[0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]
     [0.2 0.2 0.2 0.2 0.2]]
```

</td>
</tr>
</table>

#### C++ API

##### change namespace from `mindspore::api` to `mindspore` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
namespace ms = mindspore::api;
```

</td>
<td>

```c++
namespace ms = mindspore;
```

</td>
</tr>
</table>

##### `Context` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Context::Instance().SetDeviceTarget(ms::kDeviceTypeAscend310).SetDeviceID(0);
```

</td>
<td>

```c++
ms::GlobalContext::SetGlobalDeviceTarget(ms::kDeviceTypeAscend310);
ms::GlobalContext::SetGlobalDeviceID(0);
```

</td>
</tr>
</table>

##### rename `Tensor` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Tensor a;
```

</td>
<td>

```c++
ms::MSTensor a;
```

</td>
</tr>
</table>

##### `Model` move setting of model options from `Build` to ctor `Model` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
ms::Model model(graph_cell);
model.Build(model_options);
```

</td>
<td>

```c++
ms::Model model(graph_cell, model_context);
model.Build();
```

</td>
</tr>
</table>

##### `Model` modify `GetInputsInfo`, `GetOutputsInfo` to `GetInputs`, `GetOutputs` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<std::string> names;
std::vector<ms::DataType> types;
std::vector<std::vector<int64_t>> shapes;
std::vector<size_t> mem_sizes;
model.GetInputsInfo(&names, &types, &shapes, &mem_sizes);
std::cout << "Input 0 name: " << names[0] << std::endl;
```

</td>
<td>

```c++
auto inputs = model.GetInputs();
std::cout << "Input 0 name: " << inputs[0].Name() << std::endl;
```

</td>
</tr>
</table>

##### `Model` modify `Predict` parameters type from `Buffer` to `MSTensor` ([!11574](https://gitee.com/mindspore/mindspore/pulls/11574))

<table>
<tr>
<td style="text-align:center"> 1.1.0 </td> <td style="text-align:center"> 1.1.1 </td>
</tr>
<tr>
<td>

```c++
std::vector<ms::Buffer> inputs;
std::vector<ms::Buffer> outputs;
model.Predict(inputs, &outputs);
```

</td>
<td>

```c++
std::vector<ms::MSTensor> inputs;
std::vector<ms::MSTensor> outputs;
model.Predict(inputs, &outputs);
```

</td>
</tr>
</table>

### Deprecations

#### Python API

##### `ops.SpaceToBatch`, `ops.BatchToSpace` are deprecated in favor of `ops.SpaceToBatchND`, `ops.BatchToSpaceND`([!11527](https://gitee.com/mindspore/mindspore/pulls/11527))

The `ops.SpaceToBatchND`, `ops.BatchToSpaceND` are more general and have same behavior as `ops.SpaceToBatch`, `ops.BatchToSpace` when `block_shape` is a int.

##### `ops.DepthwiseConv2dNative` is deprecated in favor of `nn.Conv2D`([!11702](https://gitee.com/mindspore/mindspore/pulls/11702))

The `ops.DepthwiseConv2dNative` is only supported by Ascend, it is recommended to directly use `nn.Conv2D`. If `group` is equal to `in_ channels` and `out_channels`, the 2D convolution layer is also a 2D depthwise convolution layer.

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

# MindSpore 1.1.0 Release Notes

## MindSpore

### Major Features and Improvements

#### NewModels

- [STABLE] GNMT v2: similar to the model described in Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation, which is mainly used for corpus translation, on WMT Englis-German dataset.(Ascend)
- [STABLE] MaskRCNN: a conceptually simple, flexible, and general framework for object instance segmentation on COCO2017 dataset.(Ascend)
- [STABLE] YOLOv4: a state-of-the-art detector which is faster and more accurate than all available alternative detectors on MS COCO dataset.(Ascend)
- [STABLE] Openpose: proposes a bottom-up human attitude estimation algorithm using Part Affinity Fields on COCO2017 dataset.(Ascend)
- [STABLE] CNN-CTC: proposes three major contributions to addresses scene text recognition (STR) on MJSynth and SynthText dataset.(Ascend)
- [STABLE] CenterFace: a practical anchor-free face detection and alignment method for edge devices on WiderFace dataset.(Ascend)
- [STABLE] ShuffleNetV2:  a much faster and more accurate network than the previous networks on ImageNet 2012 dataset.(GPU)
- [STABLE] EfficientNet-B0: a new scaling method that uniformly scales all dimensions of depth/width/resolution using a simple yet highly effective compound coefficient on ImageNet 2012 dataset.(GPU)
- [BETA] SSD-GhostNet: based on an Ghost module structure which generate more features from cheap operations on Oxford-IIIT Pet dataset.(Ascend)
- [BETA] DS-CNN:  Depthwise separable convolutional neural network on Speech commands dataset.(Ascend)
- [BETA] DeepPotentialH2O: A neural network model for molecular dynamics simulations. (Ascend)
- [BETA] GOMO: A classical numerical method called GOMO for ocean simulation. (GPU)

#### FrontEnd

- [STABLE] Refactor the MINDIR to support 310 inference(Ascend).
- [STABLE] The execution backend of sparse operations in optimizer can be set through 'target'. (Ascend/GPU/CPU)
- [STABLE] Support saving specified network to checkpoint and filtering parameters according to prefix when load checkpoint. (Ascend/GPU/CPU)
- [STABLE] Allow users choose whether to load parameter into network strictly.(Ascend/GPU/CPU)
- [STABLE] Before training, in graph mode, in order to have the same network initialization parameter values ​​for all devices, broadcast the parameters on device 0 to other devices. (Ascend/GPU)
- [STABLE] Support if by if of control flow subgraph. (Ascend/GPU)
- [STABLE] Support the judgment that whether a tensor is in a list. (Ascend/GPU/CPU)
- [STABLE] Support to get a value by using the corresponding key in a dictionary in the network; Support to get keys and values of a dictionary in the network. (Ascend/GPU/CPU)
- [STABLE] Support Tensor in enumerate. (Ascend/GPU/CPU)
- [STABLE] Support multilevel index assignment. (Ascend/GPU/CPU)
- [STABLE] Support the 'expand_as','view','abs','mean' method of Tensor. (Ascend/GPU/CPU)
- [STABLE] Support ResizeBilinear operation transfer ratio. (Ascend)
- [STABLE] nn.Matmul supports matrix-vector product and  batched matrix multiply. (Ascend/GPU)
- [STABLE] nn.Dense supports input tensor whose dimension can be greater than 2. (Ascend/GPU)
- [BETA] Support higher order differentiation for partial operators.(CPU/GPU/Ascend)
- [STABLE] Support Tensor Augassign.(Ascend/GPU)
- [BETA] Support 22 numpy native interfaces.

#### Auto Parallel

- [STABLE] Support parallel optimizer with weight shard. (Ascend/GPU)
- [STABLE] Support distributed operators: element-wise series, UnsortedSegmentSum, UnsortedSegmentMin, Split, BroadcastTo and Unique etc. (Ascend/GPU)
- [STABLE] Support distributed model prediction. (Ascend/GPU)
- [STABLE] Support auto mixed precision level "O2" in auto and semi auto parallel mode. (Ascend/GPU)
- [STABLE] Add MultiFieldEmbeddingLookup high-level interface. (Ascend/GPU)

#### Executor

- [STABLE] ResNet50 performance optimize. (GPU)
- [STABLE] Support modelzoo net in PyNative mode(Ascend 29, GPU 23, CPU 2).(Ascend/GPU/CPU)
- [STABLE] Support PyNative mode on CPU.(CPU)
- [STABLE] Optimize performance in PyNative mode.(Ascend/GPU/CPU)
- [STABLE] Support Safe Optimized Memory Allocation Solver (SOMAS) on Ascend to improve the memory-reuse, the batch size of Bert large model (128 sequence length) is increased from 160 to 208.(Ascend)
- [BETA] Support second order differentiation in PyNative mode.(Ascend/GPU)
- [DEMO] Add distributed trainning in PyNative mode.(Ascend/GPU)

#### MDP

- [STABLE]  Add new operators for Ascend and GPU: IGamma, LGamma, DiGamma;
- [STABLE]  Add new distributions for Ascend and GPU: LogNormal, and Logistic;
- [BETA]  Add new distributions for Ascend only: Gumbel, Cauchy, Gamma, Beta, and Poisson; Add Categorical distribution for GPU;
- [STABLE]  Add new bijectors for Ascend and GPU: GumbelCDF, Invert;
- [STABLE]  Add Bayesian layer realized by local reparameterization method for Ascend and GPU;
- [STABLE]  Add Anomaly Detection Toolbox based on VAE for Ascend and GPU.

#### DataSet

- [STABLE] Support single node multi-p distributed cache data sharing
- [STABLE] Support GPU profiling with data processing
- [STABLE] Support YOLOV3 dynamic shape in sink mode with dataset
- [STABLE] Support unique processing in the data processing pipeline
- [STABLE] Python layer parameter verification error information unified

### API Change

#### Backwards Incompatible Change

##### Python API

###### Delete shape and dtype of class Initializer ([!7373](https://gitee.com/mindspore/mindspore/pulls/7373/files))

Delete shape and dtype attributes of Initializer class.

###### Modify the return type of initializer ([!7373](https://gitee.com/mindspore/mindspore/pulls/7373/files))

Previously, the return type of initializer function may be string, number, instance of class Tensor or subclass of class Initializer.

After modification, initializer function will return instance of class MetaTensor, class Tensor or subclass of class Initializer.

Noted that the MetaTensor is forbidden to initialize parameters, so we recommend that use str, number or subclass of Initializer for parameters initialization rather than the initializer functions.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore import dtype as mstype
>>>
>>> def conv3x3(in_channels, out_channels)
>>>   weight = initializer('XavierUniform', shape=(3, 2, 32, 32), dtype=mstype.float32)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init=weight, has_bias=False, pad_mode="same")
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> #1) using string
>>> def conv3x3(in_channels, out_channels)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init='XavierUniform', has_bias=False, pad_mode="same")
>>>
>>> #2) using subclass of class Initializer
>>> def conv3x3(in_channels, out_channels)
>>>   return nn.Conv2d(in_channels, out_channels, weight_init=XavierUniform(), has_bias=False, pad_mode="same")
```

</td>
</tr>
</table>

Advantages:
After modification, we can use the same instance of Initializer to initialize parameters of different shapes, which was not allowed before.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> weight_init_1 = XavierUniform(gain=1.1)
>>> conv1 = nn.Conv2d(3, 6, weight_init=weight_init_1)
>>> weight_init_2 = XavierUniform(gain=1.1)
>>> conv2 = nn.Conv2d(6, 10, weight_init=weight_init_2)
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> from mindspore.common import initializer
>>> from mindspore.common.initializer import XavierUniform
>>>
>>> weight_init = XavierUniform(gain=1.1)
>>> conv1 = nn.Conv2d(3, 6, weight_init=weight_init)
>>> conv2 = nn.Conv2d(6, 10, weight_init=weight_init)
```

</td>
</tr>
</table>

###### Modify get_seed function ([!7429](https://gitee.com/mindspore/mindspore/pulls/7429/files))

Modify get_seed function implementation

Previously, if seed is not set, the value of seed is default, parameters initialized by the normal function are the same every time.

After modification, if seed is not set, the value of seed is generated randomly, the initialized parameters change according to the random seed.

If you want to fix the initial value of parameters, we suggest to set seed.

```python
>>> from mindspore.common import set_seed
>>> set_seed(1)
```

###### `nn.LinSpace` ([!9494](https://gitee.com/mindspore/mindspore/pulls/9494)) has been removed and modify `ops.LinSpace` ([!8920](https://gitee.com/mindspore/mindspore/pulls/8920))

The `nn.LinSpace` interface only support passing the value by args previously. For the convenience, we provided enhancive `ops.LinSpace` interface, which support passing the value by the inputs at the latest version. So there is no need for `nn.LinSpace`.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore import nn
>>>
>>> start = 1
>>> stop = 10
>>> num = 5
>>> linspace = nn.LinSpace(start, stop, num)
>>> output = linspace()
```

</td>
<td>

```python
>>> import mindspore
>>> from mindspore import Tensor
>>> from mindspore import ops
>>>
>>> linspace = ops.LinSpace()
>>> start = Tensor(1, mindspore.float32)
>>> stop = Tensor(10, mindspore.float32)
>>> num = 5
>>> output = linspace(start, stop, num)
```

</td>
</tr>
</table>

###### Parts of `Optimizer` add target interface ([!6760](https://gitee.com/mindspore/mindspore/pulls/6760/files))

The usage of the sparse optimizer is changed.

The target interface is used to set the execution backend of the sparse operator.

The add_primitive_attr interface is no longer allowed.

The following optimizers add the target interface:  Adam, FTRL, LazyAdam, ProximalAdagrad

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn import Adam
>>>
>>> net = LeNet5()
>>> optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
>>> optimizer.sparse_opt.add_prim_attr("primitive_target", "CPU")
```

</td>
<td>

```python
>>> from mindspore.nn import Adam
>>>
>>> net = LeNet5()
>>> optimizer = Adam(filter(lambda x: x.requires_grad, net.get_parameters()))
>>> optimizer.target = 'CPU'
```

</td>
</tr>
</table>

###### `export` Modify the input parameters and export's file name ([!7385](https://gitee.com/mindspore/mindspore/pulls/7385)， [!9057](https://gitee.com/mindspore/mindspore/pulls/9057/files))

Export the MindSpore prediction model to a file in the specified format.

The reference includes: `net`, `*inputs`, `file_name`, `file_format`, `**kwargs`.

Input parameters can be input according to specific export requirements.

Add the file name extension based on the format.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.train.quant import quant
>>>
>>> network = LeNetQuant()
>>> inputs = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
>>> quant.export(network, inputs, file_name="lenet_quant.mindir", file_format='MINDIR')
lenet_quant.mindir
```

</td>
<td>

```python
>>> from mindspore import export
>>>
>>> network = LeNetQuant()
>>> inputs = Tensor(np.ones([1, 1, 32, 32]), mindspore.float32)
>>> export(network, inputs, file_name="lenet_quant", file_format='MINDIR', quant_mode='AUTO')
lenet_quant.mindir
```

</td>
</tr>
</table>

###### `Dense`, `Conv2dBnAct`, `DenseBnAct`, `DenseQuant` support setting the activation attribute as an instance of a class derived from `nn.Cell` or `Primtive` ([!7581](https://gitee.com/mindspore/mindspore/pulls/7581))

activation (Union[str, Cell, Primitive]): activate function applied to the output of the fully connected layer

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn as nn
>>>
>>> dense = nn.Dense(1, 1, activation='relu')
```

</td>
<td>

```python
>>> import mindspore.nn as nn
>>> import mindspore.ops as ops
>>>
>>> dense = nn.Dense(1, 1, activation=nn.ReLU())
>>> dense = nn.Dense(1, 1, activation=ops.ReLU())
```

</td>
</tr>
</table>

###### `tensor.dim()`, `tensor.size()` has been renamed to `tensor.ndim`, `tensor.size` ([!10175](https://gitee.com/mindspore/mindspore/pulls/10175))

Previously, tensor.size() and tensor.dim() were used for checking the total number of elements/dimensions in the tensor.
However, from a user's perspective, tensor.size and tensor.ndim (methods -> properties) are better choices, since they follow the numpy naming convention.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore import Tensor
>>>
>>> Tensor((1,2,3)).size()
>>> Tensor((1,2,3)).dim()
```

</td>
<td>

```python
>>> from mindspore import Tensor
>>>
>>> Tensor((1,2,3)).size
>>> Tensor((1,2,3)).ndim
```

</td>
</tr>
</table>

###### `EmbeddingLookup` add a config in the interface: sparse ([!8202](https://gitee.com/mindspore/mindspore/pulls/8202))

sparse (bool): Using sparse mode. When 'target' is set to 'CPU', 'sparse' has to be true. Default: True.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn import EmbeddingLookup
>>>
>>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
>>> result = EmbeddingLookup(4,2)(input_indices)
>>> print(result.shape)
(2, 2, 2)
```

</td>
<td>

```python
>>> from mindspore.nn import EmbeddingLookup
>>>
>>> input_indices = Tensor(np.array([[1, 0], [3, 2]]), mindspore.int32)
>>> result = EmbeddingLookup(4,2)(input_indices, sparse=False)
>>> print(result.shape)
(2, 2, 2)
```

</td>
</tr>
</table>

###### `nn.probability.bijector` change types of attributes from (int, float) to (float, list, numpy.ndarray, Tensor) ([!8191](https://gitee.com/mindspore/mindspore/pulls/8191))

Attributes Type change: (int, float) -> (float, list, numpy.ndarray, Tensor).
Int type is not supported anymore. Parameters of all bijectors should be type float, list, numpy.ndarray or Tensor.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> power = 2
>>> bijector = msb.PowerTransform(power=power)
```

</td>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> power = 2.0
>>> bijector = msb.PowerTransform(power=power)
```

</td>
</tr>
</table>

###### `nn.probability.bijector.GumbelCDF` remove a attribute in the interface: dtype ([!8191](https://gitee.com/mindspore/mindspore/pulls/8191))

dtype is removed from GumbelCDF and is no longer an argument of the class.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>> from mindspore import dtype as mstype
>>>
>>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0, dtype=mstype.float32)
```

</td>
<td>

```python
>>> import mindspore.nn.probability.bijector as msb
>>>
>>> bijector = msb.GumbelCDF(loc=0.0, scale=1.0)
```

</td>
</tr>
</table>

###### `nn.layer.combined.Conv2dBnAct`, `nn.layer.combined.DenseBnAct` move from nn.layer.quant to nn.layer.combined ([!8187](https://gitee.com/mindspore/mindspore/pulls/8187))

Previously Conv2dBnAct and DenseBnAct are in nn.layer.quant, since they are not quant cells, now they are moved to nn.layer.combined. If you import Conv2dBnAct, DenseBnAct from mindspore.nn, then your code doesn't need any change.

<table>
<tr>
<td style="text-align:center"> 1.0.1 </td> <td style="text-align:center"> 1.1.0 </td>
</tr>
<tr>
<td>

```python
>>> from mindspore.nn.layer.quant import Conv2dBnAct, DenseBnAct
```

</td>
<td>

```python
>>> from mindspore.nn import Conv2dBnAct, DenseBnAct
```

</td>
</tr>
</table>

###### `nn.layer.conv.Conv2D`, `nn.layer.quant.Conv2dBnFoldQuant`, `nn.layer.quant.Conv2dBnWithoutFoldQuant` change weight shape when group > 1 in Ascend platform ([!9723](https://gitee.com/mindspore/mindspore/pulls/9723))

In Ascend platform, if group > 1, the weight shape of Conv2D change from [in_channels//group, out_channels, kernel_size, kernel_size] to [out_channels, in_channels//group, kernel_size, kernel_size]. Previously, checkpoints of the networks are used, which use Conv2D with group > 1, such as MobileNet, can not be directly used now, need to transpose the first and second axis of the weight.

### Bug fixes

#### FrontEnd

- [STABLE] Fix the problem of the cse optimization in the situation of control flow. (Ascend/GPU)

#### Auto Parallel

- [STABLE] Resolve the restriction: input and output layouts of Reshape are restricted in tensor redistribution. (Ascend/GPU)
- [STABLE] Resolve the restriction: output strategy should be data parallel in model evaluation. (Ascend/GPU)

#### Executor

- [STABLE] Fix fusion operator compilation cache. (Ascend)
- [STABLE] Fix compilation error of dynamic shape operator. (Ascend)
- [STABLE] Fix bug of pynative cannot insert transdata of node output when node should be spilted in the backend opt.(Ascend)
- [STABLE] Fix the bug of TensorMove and memcpy_async merge to one after backend cse pass (Ascend)

#### DataSet

- [STABLE] Fix cache server hang on RequestFreeTag. (Ascend/GPU/CPU)
- [STABLE] Fix hung when use pyfunc multi-processing. (Ascend/GPU/CPU)
- [STABLE] Fix add multiple parent nodes to tree node cause core dump. (Ascend/GPU/CPU)

## MindSpore Lite

### Major Features and Improvements

#### Converter and runtime

1. Support dynamic shape in MindSpore Lite Converter.
2. Optimize sub-graph mechanism by dynamically splitting the entire graph into multiple subgraphs based on the operator supported, backend hardware and user configuration.
3. Support TensorList and TensorList operators such as TensorListFromTensor, TensorListGetItem and so on.
4. Support BatchMatMul fusion and LSTM fusion in MindSpore Lite Converter.
5. Support converting model and run inference on Windows operator system.
6. Support Model(.ms) visualization on Netron.
7. Support Tensorflow model in MindSpore Lite Converter
8. Add 86 converter parsers.
9. Convert aware training model without user’s awareness
10. Support scalar tensor in MindSpore Lite Converter and Runtime
11. Support NPU backend on HUAWEI Kirin SoC.[BETA]
12. Merge timeprofiler into benchmark

#### CPU backend optimization

1. Add 50+ new operators, including new Op type(like Adder, Gru).
2. Enhanced performance on armv8.2 supported platform. For example, utilizing sdot instruction more efficiently.
3. Optimize all operators(fp32, fp16, int8) by implementing multi-thread, SIMD tech as much as possible. Model inference time can reduce at least 20% after these optimizations.
4. Extending to support operators for x86_64 platform based on SSE/AVX instruction set.

#### OpenCL backend

1. Add new ops: add 10+ ops, total 58 ops;
2. Performance optimization: by memory layout optimize, Winograd Convolution select strategyoptimize, SIMT local size optimize, local cache optimize,  GPU performance improvement up to 20+% vs MSLITE Version1.0
3. Add Online Graph optimzation: by fusion Convolution/Matmul/Fullconnection and add/mul/pad/reshape, improve performance up to 50+% for some networks;
4. Add auto tuning: by online tuning in the graph compilation phase, optimize performance up to 10%;
5. Add weight quant: support weight quant
6. Add opencl kernel binary cache: improve Initialization time .

#### Post quantization

MindSpore Lite supports both weight quantization and full quantization. Currently, Weights can be quantized into 1 ~ 16 bits according to user configuration. In internal testing, quantization of networks, such as classification, detection, segmentation and transformer are well supported. To ensure high accuracy of quantized models, MindSpore Lite uses a pipeline quantization method. In the first phase, the weight and activation value are quantized using linear quantization methods, such as MIN-MAX. In the second phase, the quantization error is analyzed, and uses statistical methods to compensate loss caused by fp32 quantization to a fixed point such as Int8 to quantized models. The features of Post-training quantization are:

1. perchannel asymmetric quantization for weights, such as MAX_MIN and KMEANS
2. Perlayer symmetric quantization for activation, such as KL and MAX_MIN.
3. perlayer asymmetrical quantization for activation, such as, RemoveOutlier.
4. accuracy loss compensation, such as BiasCorrection

| mobilenet_v2   | ACC (ImageNet)  |
|---|---|
| FP32  | 71.56%  |
|A8W8   | 71.16%  |
| A8W8(without BiasCorrection)  | 70.74% |
| A8W7  | 71.06%  |
| A7W7  | 70.78%  |

The above table uses the mobilenet_v2 model from TF official website. Using MindSpore Lite quantization, the precision of A8W8 (8-bit activation value quantization and 8-bit weight quantization) decreases from 0.82% to 0.4% after accuracy loss compensation, for 7-bit quantization, the precision loss is still no more than 1%.

#### Training on Device

Within MindSpore 1.1 release, the MindSpore Lite provides the following Training-on-Device (ToD) capabilities:

1. Learning from scratch and Transfer Learning strategies are supported
2. MindSpore based models can be converted and used in training on the device. (Third-party models such as TensorFlow and PyTorch for now cannot be directly imported to the framework)
3. Grad operations are supported for more than 30 operators such as Dense layers, Convolutions and Batch Normalizations. Momentum, SGD, and ADAM optimizers are supported.
4. Supports networks such as LeNet, Alexnet, Resnet, MobileNetV1/V2/V3, and EffectiveNet, and provides complete model loading, conversion, and Python training scripts on the device side.

The MindSpore Lite ToD framework is already in use in the newest Huawei Smart TV, providing a unique and personalized user experience as a family entertainment center.

### API Change

#### API Incompatible Change

##### C++ API

- [Modify] Context now support multi-context configuration.(Context.h)
- [Modify] Callback is move from lite_session.h into ms_tensor.h.
- [Modify] GetInputsByName in lite_session.h is changed into GetInputsByTensorName
- [Add] add static LiteSession *CreateSession(const char*model_buf, size_t size, const lite::Context *context) in lite_session.h
- [Add] add GetErrorInfo interface returning error message in errorcode.h
- [Delete] Remove model_generated.h, ops_generated.h and headers of FlatBuffers library from interfaces

##### Java API

- [Add] Implement JNI layer and add Java api for CPU and GPU backend

#### Deprecations

##### C++ API

Deprecate Interface GetOutputsByNodeName

### Bug fixes

- [BUGFIX] Fix the bug in sub-graph segmentation
- [BUGFIX] Fix the bug in Tensor getitem in which the ellipsis matches the wrong dim-size.
- [BUGFIX] Fix the bug that activation modification after defining Dense will not take effect.

## Contributors

Thanks goes to these wonderful people:

zhouyifengCode, huqi, JulyAi, damon0626, chenbo116, rmdyh, davidmc, gray0v0, doitH, Gogery, zymaa, xinyunfan

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenbo116, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, damon0626, danish, Danish, davidmc, dayschan, doitH, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, Gogery, gongdaguo, gray0v0, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huqi, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, JulyAi, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, rmdyh, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xinyunfan, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, zhouyifengCode, Zichun, Zirui, Ziyan, zjun, ZPaC, zymaa

Contributions of any kind are welcome!

# MindSpore 1.0.0 Release Notes

## Major Features and Improvements

### MindSpore Training and Inference Framework

#### Ascend 910

- New models
    - DenseNet121: a dense convolutional neural network, which connects each layer to every other layer in a feed-forward fashion for object recognition on ImageNet dataset.
    - UNet2D-Medical: Unet Medical model for 2D image segmentation, Convolutional Networks for Biomedical Image Segmentation on ISBI Challenge database.
- Frontend and user interface
    - Second-Order Optimization
        - Enable second-order optimization for Bert on Ascend 910, which can achieve a masked lm accuracy of 71.3% in 800 seconds using 8 Ascend 910 (Bert-Large @MLPerf v0.7 dataset).
    - New GNN model BGCF
        - Bayesian Graph Convolutional Filtering network which naturally incorporate the uncertainty in the user-item interaction graph shows excellent recommendation performance on Amazon-Beauty dataset.
    - Add append interface for SequentialCell.
    - Add a level `auto` for AMP.
- Executor and performance optimization
    - Support quantitative network (Resnet50 & YoloV3 & MobileNetV2).
    - Project ease of use optimization: project compilation time optimization, CMakelist regularization, cudnn, cuda independent compilation and installation independent.
- Data processing, augmentation, and save format
    - Support GeneratorDataset return string type

#### Other Hardware Support

- GPU platform
    - Enable second-order optimization for resnet50 on GPU, which achieve 30% improvement on training time compared to SGD with Momentum (Resnet50 @ImageNet).

#### User interfaces change log

- Remove global object GradOperation in Autodiff([!5011](https://gitee.com/mindspore/mindspore/pulls/5011))
- Remove useless attribute 'name' in Autodiff([!5172](https://gitee.com/mindspore/mindspore/pulls/5172))
- Rectification distributed init([!5350](https://gitee.com/mindspore/mindspore/pulls/5350))
- Move the setting of ParalleMode from train.parallel_utils to context([!5351](https://gitee.com/mindspore/mindspore/pulls/5351))
- Modification of save_checkpoint([!5482](https://gitee.com/mindspore/mindspore/pulls/5482))
- Wrap numpy random seed into an api([!5634](https://gitee.com/mindspore/mindspore/pulls/5634))
- Delete enable_fused_layernorm in some modelzoo scripts([!5665](https://gitee.com/mindspore/mindspore/pulls/5665))
- Move 'multi-subgraphs' interface to internal([!5696](https://gitee.com/mindspore/mindspore/pulls/5696))
- Rename mirror_mean to gradient_mean([!5700](https://gitee.com/mindspore/mindspore/pulls/5700))
- Remove default value of 'group' of DepthWiseConv2d([!5865](https://gitee.com/mindspore/mindspore/pulls/5865))
- Modify interface for function and remove duplicated def([!5958](https://gitee.com/mindspore/mindspore/pulls/5958))
- Unify Conv2d and DepthwiseConv2d([!5916](https://gitee.com/mindspore/mindspore/pulls/5916))
- Modification of SoftmaxCrossEntropyWithLogits([!5502](https://gitee.com/mindspore/mindspore/pulls/5502))
- Change API set_strategy() to shard()([!5991](https://gitee.com/mindspore/mindspore/pulls/5991))
- Move batch_size from bert_cfg_cfg to cfg([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
- Remove unused parameters from SummaryRecord __init__([!5548](https://gitee.com/mindspore/mindspore/pulls/5548))
- remove sens parameter of TrainOneStepWithLossScaleCell([!5753](https://gitee.com/mindspore/mindspore/pulls/5753))
- optimize the TrainOneStepCell for user's define([!6159](https://gitee.com/mindspore/mindspore/pulls/6159))
- delete seed0 and seed1 of nn.Dropout([!5735](https://gitee.com/mindspore/mindspore/pulls/5735))
- delete DataWrapper([!6101](https://gitee.com/mindspore/mindspore/pulls/6101))
- LSTM API optimization([!6374](https://gitee.com/mindspore/mindspore/pulls/6374))
- Merge P\C\F of ops([!5645](https://gitee.com/mindspore/mindspore/pulls/5645))
- delete SoftmaxCrossEntropyExpand interface([!6607](https://gitee.com/mindspore/mindspore/pulls/6607))
- Adjust GroupNorm interface([!6329](https://gitee.com/mindspore/mindspore/pulls/6329))
- Modify init interface to internal interface([!6651](https://gitee.com/mindspore/mindspore/pulls/6651))
- Log optimization([!5842](https://gitee.com/mindspore/mindspore/pulls/5842))
- Remove useless API dataset.set_dataset_size（[!5806](https://gitee.com/mindspore/mindspore/pulls/5806))
- Some of Dataset API add usage parameter（[!5605](https://gitee.com/mindspore/mindspore/pulls/5605))
- Change the import path, such as from mindspore.dataset.transforms.vision to mindspore.dataset.vision.transforms（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Rename ImageFolderDatasetV2 to ImageFolderDataset（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Dataset.map parameter optimization（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Add new api dataset.get_col_names（[!5384](https://gitee.com/mindspore/mindspore/pulls/5384))
- Remove useless API MindRecord finish（[!5580](https://gitee.com/mindspore/mindspore/pulls/5580))

### MindSpore Lite

- Converter
    - Add 6 TFLite op, 7 Caffe op, 1 ONNX op.
    - Add support for Windows.
    - Support parallel inference of multiple sessions to adapt to more scenarios
    - Support 8bits only weight-quantization, most main-stream models has small accuracy loss (less than 0.5%) when compared to non-qunantized fp32 model.

- CPU & GPU
    - Add 20 CPU ops，include FP32, int8/uint8, FP16 and int32 ops.
    - Add supporting FP16 for GPU, add 14 GPU ops include FP32/FP16.
    - Add Buffer/Image2D transform op for GPU
    - Performance optimization for CPU ops focus on ARM32.
    - Performance optimization for GPU Convolution using winograd.

- Tool & example
    - Add object detection Android Demo.

## Bugfixes

- Models
    - fix the constant folding problem in multiply.([!6092](https://gitee.com/mindspore/mindspore/pulls/6092))
    - move batch_size from bert_net_cfg to cfg in bert scripts.([!6233](https://gitee.com/mindspore/mindspore/pulls/6233))
    - modify the checkpoint file path.([!6137](https://gitee.com/mindspore/mindspore/pulls/6137))
- Python API
    - fix semi auto parallel parameter of reshape has another user([!5722](https://gitee.com/mindspore/mindspore/pulls/5722))
    - raise ValueError when call hook function in graph mode([!5831](https://gitee.com/mindspore/mindspore/pulls/5831))
- Executor
    - fix pynative mode to build temporary nn objects.（[!6189](https://gitee.com/mindspore/mindspore/pulls/6189))
    - fix the accuracy problem of multiple inputs of multi-card communication operator broadcast.([!6522](https://gitee.com/mindspore/mindspore/pulls/5622))
    - fix the problem that the sample distribution interface categorical does not support graph mode.([!5772](https://gitee.com/mindspore/mindspore/pulls/5772))
    - fix the random seed failure problem of the polynomial downsampling distribution operator.([!5948](https://gitee.com/mindspore/mindspore/pulls/5948))
    - fix unnecessary address binding issues in GPU heterogeneous scenarios.([!6232](https://gitee.com/mindspore/mindspore/pulls/6232))
- GPU platform
    - fix for kernel resource leak([!5315](https://gitee.com/mindspore/mindspore/pulls/5315))
    - fix for insufficient memory for continuous unit test running([!5617](https://gitee.com/mindspore/mindspore/pulls/5617))
    - fix for the memory leak in the sparse slicer([!5578](https://gitee.com/mindspore/mindspore/pulls/5578))
- Data processing
    - fix hang when use pyfunc([!6346](https://gitee.com/mindspore/mindspore/pulls/6346))
    - fix GPU device queue does not release GIL during resource clean up([!5964](https://gitee.com/mindspore/mindspore/pulls/5964))
    - fix hang if scripte exit unnormally([!6441](https://gitee.com/mindspore/mindspore/pulls/6441))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, AGroupofProbiotocs, anthonyaje, anzhengqi, askmiao, baihuawei, baiyangfan, bai-yangfan, bingyaweng, BowenK, buxue, caifubi, CaoJian, caojian05, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chenzomi, chenzupeng, chujinjin, cj, cjh9368, Corleone, danish, Danish, dayschan, eric, Eric, fary86, fuzhiye, Gaoxiong, gengdongjie, gongdaguo, gukecai, guoqi, gzhcv, hangq, hanhuifeng2020, Harshvardhan, He, heleiwang, hexia, Hoai, HuangBingjian, huangdongrun, huanghui, huangxinjing, huzhifeng, hwjiaorui, Jesse, jianghui58, jiangzhiwen, Jiaqi, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, jzg, kai00, kingfo, kingxian, kpy, kswang, laiyongqiang, leonwanghui, Li, liangchenghui, liangzelang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, linqingke, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuyang_655, liuzhongkai, Lixia, lixian, liyanliu, liyong, lizhenyu, luoyang, lvchangquan, lvliang, lz, mahdi, Mahdi, maning202007, Margaret_wangrui, mayang, mengyuanli, nhussain, ougongchang, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, Pengyongrong, qianlong, r1chardf1d0, riemann_penn, root, Sheng, shenwei41, simson, Simson, Su, sunsuodong, tao_yunhao, tinazhang, VectorSL, , Wan, wandongdong, wangdongxu, wangmin, wangnan39@huawei.com, wangyue01, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, Xiaoda, xiefangqi, xuanyue, xulei2020, Xun, xuyongfei, yanghaitao, yanghaitao1, yanghaoran, YangLuo, yangruoqi713, yankai, yanzhenxiang2020, yao_yf, yepei6, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zengzitao, Zhang, zhanghaibo5@huawei.com, zhanghuiyao, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaojichen, zhaoting, zhaozhenlong, zhengjun10, zhoufeng, zhousiyi, zhouyaqiang, Zichun, Zirui, Ziyan, zjun, ZPaC

Contributions of any kind are welcome!

# MindSpore 0.7.0-beta Release Notes

## Major Features and Improvements

### MindSpore Training and Inference Framework

#### Ascend 910

- New models
    - TinyBert: a smaller and faster version of BERT using transformer distillation for natural language understanding on GLUE benchmark.
    - SE-ResNet50: add Squeeze-and-Excitation blocks(SE-Blocks) to the resnet50 network to improve channel interdependencies for image classification on ImageNet 2012 dataset.
    - Inception V3: the third version of Inception convolutional architectures for image classification on ImageNet 2012 dataset.
- Frontend and user interface
    - Embedding operator high-level packaging to support segmented by field for Wide&Deep.
    - Load multi-node checkpoint into single-process to support host-device hybrid inference.
    - Support Concat/Tile/Strideslice distributed operators.
    - Support cumulative gradient and batch training split.
    - Support variable parameter input for Cell object.
    - Parameter mixed calculation optimization for pynative mode.
    - Deep Probabilistic Programming
        - Support statistical distributions classes used to generate stochastic tensors.
        - Support probabilistic inference algorithms.
        - Support BNN layers used to construct BNN in Graph mode.
        - Support interfaces for the transformation between BNN and DNN in Graph mode.
        - Support uncertainty estimation to estimate epistemic uncertainty and aleatoric uncertainty.
    - User interfaces change log
        - change base class of parameter([!3473](https://gitee.com/mindspore/mindspore/pulls/3473))
        - change binary to mindir([!4258](https://gitee.com/mindspore/mindspore/pulls/4258))
        - change export from geir to air([!4269](https://gitee.com/mindspore/mindspore/pulls/4269))
        - Init parameter data by default([!3967](https://gitee.com/mindspore/mindspore/pulls/3967))
        - change IndexedSlices to RowTensor([!4031](https://gitee.com/mindspore/mindspore/pulls/4031))
        - Must set or change parallel mode before any Initializer created([!4801](https://gitee.com/mindspore/mindspore/pulls/4801))
- Executor and performance optimization
    - MindSpore graph compilation process performance improved by 20%.
    - Decoupling C++ and Python modules to achieve separate compilation of core modules.
- Data processing, augmentation, and save format
    - Support automatic data augmentation
    - Support GNN distributed cache in single node
    - Support ConcatDataset using distributed sampler

#### Other Hardware Support

- GPU platform
    - New model supported: VGG16, ResNet101, DeepFM.
    - Support some distributed operators in ResNet50 and Wide&Deep.
    - Support automatic parallel for Wide&Deep.
    - Support function funcs[i](*inputs) (such as switch-case).
    - Support distributed training with parameter server.
    - Support GPU operator profiling.
    - Performance optimization of the distributed training with allreduce.
    - Performance optimization of the mixed precision training.
    - Performance optimization of the pynative mode.
    - Performance optimization of the convolution operator, batch normalization operator.
- CPU platform
    - Support MobileNetV2 Re-Training: Re-train the network with different class number.

### MindSpore Lite

- Converter
    - Support third-party models, including TFLite/Caffe/ONNX.
    - Add 93 TFLite op.
    - Add 24 Caffe op.
    - Add 62 ONNX op.
    - Add 11 optimized passes, include fusion/const fold.
    - Support aware-training and Post-training quantization.
- CPU
    - Add 100+ops，support fp32, int8/uint8, FP16 ops
    - Support fast convolution algorithms: Sliding Window, Img2col + Gemm, Strassen, Winograd
    - Support assembly/neon instruction.
    - Support CPU fp16 and sdot on ARM v8.2+.
- GPU
    - Add 20+ ops for OpenCL.
    - Support image2D/buffer format.
    - Optimize online initialization time.
    - add optimized convolution1X1/3X3/depthwise/convolution_transposed for OpenCL.
- Tool & example
    - Add benchmark and TimeProfile tools.
    - Add image classification Android Demo.

## Bugfixes

- Models
    - normalize the readme file([!5410](https://gitee.com/mindspore/mindspore/pulls/5410))
    - fix a sink_size bug for transformer([!5393](https://gitee.com/mindspore/mindspore/pulls/5393))
    - fix bool type optional for resnet50([!5363](https://gitee.com/mindspore/mindspore/pulls/5363))
- Python API
    - improve interface '__bool__' for tensor([!4000](https://gitee.com/mindspore/mindspore/pulls/4000))
    - fix GPU-ResizeNearestNeighbor([!3760](https://gitee.com/mindspore/mindspore/pulls/3760))
    - fix topK multi dimension grad func([!3711](https://gitee.com/mindspore/mindspore/pulls/3711))
    - fix scatterop error msg([!3699](https://gitee.com/mindspore/mindspore/pulls/3699))
    - fix bug of cast dtype when using mix_presion in pynative mode([!3730](https://gitee.com/mindspore/mindspore/pulls/3730))
- Executor
    - fix etsnet train error when UnsegmentSum's first input shape is (1,) ([!4573](https://gitee.com/mindspore/mindspore/pulls/4573))
    - fix bug of result error in while control flow because of unsupporting for value reference ([!4103](https://gitee.com/mindspore/mindspore/pulls/4103))
    - fix bug of the output tensor does not carry device data type ([!3774](https://gitee.com/mindspore/mindspore/pulls/3774))
    - fix bug of avoiding multi attr value are eliminated in pynative mode ([!4225](https://gitee.com/mindspore/mindspore/pulls/4225))
    - fix bug of AssignAdd unable to work normally in multi-cases ([!5171](https://gitee.com/mindspore/mindspore/pulls/5171))
- GPU platform
    - improve the environment variable checking for nvcc compiler path ([!5140](https://gitee.com/mindspore/mindspore/pulls/5140))
    - fix bug of error in cast operator conversion from fp16 to fp32 ([!4147](https://gitee.com/mindspore/mindspore/pulls/4147))
    - fix bug of the array out of bound in case of make_tuple operator ([!5219](https://gitee.com/mindspore/mindspore/pulls/5219))
- Data processing and Pro
    - fix GeneratorDataset time out([!3624](https://gitee.com/mindspore/mindspore/pulls/3624))
    - fix concat operator get_dataset_size error([!4701](https://gitee.com/mindspore/mindspore/pulls/4701))
    - fixing python validator for Repeat Op([!4366](https://gitee.com/mindspore/mindspore/pulls/4366))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Adel, Alexey, andy, andy_wangrui, anthonyaje, anzhengqi, askmiao, avakh, baihuawei, bingyaweng, BowenK, buxue, caifubi, CaoJian, caozhou, Cathy, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chenzupeng, chujinjin, cjh9368, Corleone, cristoval, danish, dengyutao, eric, Eric, ervinzhang, etone-chan, fangzehua, fary86, fuzhiye, gengdongjie, genglishuai, Giancarlo, gongdaguo, gukecai, guohongzilong, GuoMengHao, hangq, hanhaocheng, hanhuifeng2020, hanjun996, Harshvardhan, He, heleiwang, hesham, hexia, Hoai, hongxing, huangdongrun, huanghui, huangxinjing, islam_amin, Jesse, jianghui58, jiangzhiwen, jin-xiulang, jinyaohui, jjfeing, John, Jonathan, jonyguo, kai00, kingfo, kpy, kswang, laiyongqiang, leilei_snow, leopz, Li, liangzelang, lianliguang, lichen_101010, lichenever, lihongkang, lilei, limingqi107, ling, lingyunli63, linqingke, lirongzhen1, liubuyu, liuwenhao4, liuxiao78, liuxiao93, liuzhongkai, Lixia, lixian, liyong, lizhenyu, looop5, luoyang, lvchangquan, lvliang, lvwenyuan, lyvette, mahdi, Mahdi, mamba_ni, maning202007, Margaret_wangrui, mayang, meixiaowei, meng_chunyang, ms_yan, nhussain, panbingao, panfengfeng, panyifeng, Payne, Peilin, peixu_ren, pengyongrong, Pengyongrong, qianlong, qujianwei, root, shenwei41, shibeiji, simson, songhonglei413, Su, sunsuodong, suteng, tao_yunhao, TFbunny, tinazhang, tom__chen, tony_liu2, tronzhang, VectorSL, wandongdong, wangdongxu, wanghua, wangmin, wangshaocong, wangzhe, wanyiming, Wei, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuweikang, wuxuejian, wuyongkang, xiefangqi, xuanyue, Xun, xutianchun, xuyongfei, yanghaitao, yangjie159, YangLuo, yangruoqi713, yangyongjie, yangzhenzhang, yankai, yao_yf, yelihua, yeyunpeng, Yi, yoni, yoonlee666, yuchaojie, yujianfeng, yuximiao, zhangxuetong, zhaizhiqiang, Zhang, zhangxinfeng3, zhangxuetong, zhangyihui, zhangz0911gm, zhanke, zhanyuan, zhaodezan, zhaoting, zhaozhenlong, zhengjun10, zhongligeng, zhoufeng, zhousiyi, zhouyaqiang, zhouyuanshen, Zichun, Zirui, zjun, zongha, ZPaC, lijiaqi, liangchenghui, wangminggui

Contributions of any kind are welcome!

# MindSpore 0.6.0-beta Release Notes

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- New models
    - There are official, research and community under modelzoo.
        - Official is maintained  with the newest APIs by MindSpore team,  MaskRCNN are added.
        - Research is uploaded by researchers for official review, and APIs may not  be updated in time.
        - Community reprints the relevant links of partner research results.
    - Hub added on the same level as modelzoo, synchronous storage of materials needed for official hub web pages which will be launched soon.
    - Support pre-trained models, few lines of code can be used to download and load pre-trained models, supporting inference or transfer learning.
- Frontend and user interface
    - Supports user side operator compilation and graph execution error rendering.
    - Uniform definition dynamic learning rate behavior in optimizers.
    - Support IndexSlice in sparse expression.
    - Support use parent construct method during construct.
    - Support asynchronous execution save checkpoint file.
    - Support implicit type conversion in pynative mode.
    - User interfaces change log
        - unform learning rate behavior in optimizers([!2755](https://gitee.com/mindspore/mindspore/pulls/2755))
        - rename operator of sparse optimizer([!3217](https://gitee.com/mindspore/mindspore/pulls/3217))
        - move profiler module from mindinsight to mindspore([!3075](https://gitee.com/mindspore/mindspore/pulls/3075))
        - VOCDataset output change to multi-columns([!3093](https://gitee.com/mindspore/mindspore/pulls/3093))
        - GetDatasize feature([!3212](https://gitee.com/mindspore/mindspore/pulls/3212))
        - dataset: modify config api([!2936](https://gitee.com/mindspore/mindspore/pulls/2936))
- Executor and performance optimization
    - Decouple C++ and python, so make the architecture more extensible.
    - Parameter Server for distributed deep learning supported.
    - Serving：a flexible service deployment framework for deep learning models.
    - Memory reuse is enhanced, and the batch size of Bert large model is increased from 96 to 160 on a single server.
- Data processing, augmentation, and save format
    - Support MindRecord save operator after  date processing
    - Support automatic fusion operator, such as decode/resize/crop
    - Support CSV dataset loading

### Other Hardware Support

- GPU platform
    - New model supported: ResNext50, WarpCTC and GoogLeNet.
    - Support hyperparametric search and data enhanced automl on GPU.
    - Support Resnet50 automatic parallel in GPU backend.

## Bugfixes

- Models
    - Improved the performance and accuracy on ResNet50([!3456](https://gitee.com/mindspore/mindspore/pulls/3456))
    - Fixed the performance test case of bert([!3486](https://gitee.com/mindspore/mindspore/pulls/3486))
- Python API
    - Fix assign used in while loop([!2720](https://gitee.com/mindspore/mindspore/pulls/2720))
    - Revert optimize the graph output of all nop node.([!2857](https://gitee.com/mindspore/mindspore/pulls/2857))
    - Print tensor as numpy.([!2859](https://gitee.com/mindspore/mindspore/pulls/2859))
    - Support weight decay for sparse optimizer([!2668](https://gitee.com/mindspore/mindspore/pulls/2668))
    - Fix BatchToSpaceND([!2741](https://gitee.com/mindspore/mindspore/pulls/2741))
    - Fixing type check mistakes of InplaceAdd and Inplace Sub ops([!2744](https://gitee.com/mindspore/mindspore/pulls/2744]))
    - Change order param only equal to group param([!2748](https://gitee.com/mindspore/mindspore/pulls/2748))
- Executor
    - The performance of graph with control flow is optimized([!2931](https://gitee.com/mindspore/mindspore/pulls/2931))
    - Fix bug of wrong number of tuple layers([!3390](https://gitee.com/mindspore/mindspore/pulls/3390))
    - Fix cpu multi graph memory exception([!3631](https://gitee.com/mindspore/mindspore/pulls/3631))
    - Enable data sync when calling operator without defining a cell([!3081](https://gitee.com/mindspore/mindspore/pulls/3081))
    - Fix argmaxwith value error in pynative mode on GPU([!3082](https://gitee.com/mindspore/mindspore/pulls/3082))
    - Fix precision error with fp16 input on pynative mode([!3196](https://gitee.com/mindspore/mindspore/pulls/3196))
- Data processing
    - Fix bug of RandomColor and RandomSharpness default parameter checking  ([!2833](https://gitee.com/mindspore/mindspore/pulls/2833))
    - Fix process hung when training and eval  ([!3469](https://gitee.com/mindspore/mindspore/pulls/3469))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).
    - Libjpeg-turbo : Update libjpeg-turbo to 2.0.4 to handle [CVE-2020-13790](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13790).

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.5.2-beta Release Notes

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- New models
    - DenseNet121: a convolution based neural network for the task of image classification on ImageNet 2012 dataset.

## Bugfixes

- Models
    - VGG16,Alexnet,GoogleNet,optimize network for better performance. ([!5539](https://gitee.com/mindspore/mindspore/pulls/5539))
    - YOLOV3, fix yolov3_darknet53 dataset bug. ([!5658](https://gitee.com/mindspore/mindspore/pulls/5658))

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.5.0-beta Release Notes

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- New models
    - ResNext50: a simple, highly modularized network architecture using aggregated resdiual transformations for image classification on ImageNet 2012 dataset.
    - MASS: a pre-training method for sequence to sequence based language generation tasks on Text Summarization and Conversational Response Generation using News Crawls 2007-2017 dataset, Gigaword corpus and Cornell movie dialog corpus.
    - Transformer: a neural network architecture for language understanding on WMT 2014 English-German dataset.
    - GCN：Graph Convolutional Networks for the task of classification of nodes in a graph on Cora and Citeseer datasets.
    - GAT：an attention-based graph neural network for node classification on Cora and CiteSeer dataset.
- Frontend and user interface
    - Support tensor value and assignment of mixed tensor index in graph mode.
    - Support tensor comparison, len operator, constexpr syntax, value and assignment of tensor index in pynative mode.
    - Support converting MindSpore IR to pb format for infer model.
    - Support print operator to write data directly on the hard disk.
    - Add the double recursive programming solution for very high speed parallel strategy search in automatic parallel.
    - User interfaces change log
        - Allow the learning rate of AdamWeightDecayDynamicLR and Lamb to be 0([!1826](https://gitee.com/mindspore/mindspore/pulls/1826))
        - Restricting the entire network input parameter is Tensor([!1967](https://gitee.com/mindspore/mindspore/pulls/1967))
        - Turn shape and dtype into attributes instead of interfaces([!1919](https://gitee.com/mindspore/mindspore/pulls/1919))
        - Delete multitypefungraph([!2116](https://gitee.com/mindspore/mindspore/pulls/2116))
        - Refactor the callback module in an encapsulated way, use _CallbackManager instead of_build_callbacks([!2236](https://gitee.com/mindspore/mindspore/pulls/2236))
        - Delete EmbeddingLookup([!2163](https://gitee.com/mindspore/mindspore/pulls/2163))
        - Checkpoint add model_type([!2517](https://gitee.com/mindspore/mindspore/pulls/2517))
- Executor and performance optimization
    - Heterogeneous execution on CPU and Ascend devices supported, and is verified in Wide&Deep model.
    - Quantitative training of MobileNetV2, Lenet and Resnet50 on Ascend-910 are supported.
    - Support new fusion architecture, which can do fusion optimization across graphs and kernels to improve execution speed.
- Data processing, augmentation, and save format
    - Support data processing pipeline performance profiling.
    - Support public dataset loading, such as CLUE and Coco.
    - Support more text processing, such as more tokenizers and vocab data.
    - Support MindRecord padded data.

### Other Hardware Support

- GPU platform
    - New model supported: Bert / Wide&Deep.
    - Support setting max device memory.
- CPU platform
    - New model supported: LSTM.

## Bugfixes

- Models
    - Bert, Move Bert from `example` to `model_zoo`, optimize network for better performance. ([!1902](https://gitee.com/mindspore/mindspore/pulls/1902))
    - VGG16, Move VGG16 from `example` to `model_zoo`, optimize network for better accuracy. ([!2645](https://gitee.com/mindspore/mindspore/pulls/2645))
    - Alexnet, modify parameter setting to improve accuracy ([!1364](https://gitee.com/mindspore/mindspore/pulls/2370))
    - Wide&Deep, Move Wide&Deep from `example` to `model_zoo`, optimize network for better performance. ([!2221](https://gitee.com/mindspore/mindspore/pulls/2221))
- Python API
    - Fix bug in auto cast([!1766](https://gitee.com/mindspore/mindspore/pulls/1766))
    - Fix bug of register_backward_hook([!2148](https://gitee.com/mindspore/mindspore/pulls/2148))
    - Fix bug of tuple args in pynative mode([!1878](https://gitee.com/mindspore/mindspore/pulls/1878))
    - Fix bug of checking numbers of arguments and graph parameters([!1701](https://gitee.com/mindspore/mindspore/pulls/1701))
- Executor
    - Fix bug of loading input data repeatedly in pynative mode([!1966](https://gitee.com/mindspore/mindspore/pulls/1966))
    - Fix bug of list cannot be used as input in pynative mode([!1765](https://gitee.com/mindspore/mindspore/pulls/1765))
    - Fix bug of kernel select ([!2103](https://gitee.com/mindspore/mindspore/pulls/2103))
    - Fix bug of pattern matching for batchnorm fusion in the case of auto mix precision.([!1851](https://gitee.com/mindspore/mindspore/pulls/1851))
    - Fix bug of generate hccl's kernel info.([!2393](https://gitee.com/mindspore/mindspore/pulls/2393))
- GPU platform
    - Fix bug of summary feature invalid([!2173](https://gitee.com/mindspore/mindspore/pulls/2173))
- Data processing
    - Fix bug of Cifar dataset reading([!2096](https://gitee.com/mindspore/mindspore/pulls/2096))
    - Fix bug of C++ behavior in RandomCropAndResize([!2026](https://gitee.com/mindspore/mindspore/pulls/2026))
    - Fix the bug of mindrecord shuffle([!2420](https://gitee.com/mindspore/mindspore/pulls/2420))
- Third party
    - Sqlite : Update sqlite to 3.32.2 to handle [CVE-2020-11656](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11656), [CVE-2020-13871](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13871), [CVE-2020-11655](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655), [CVE-2020-9327](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-9327), [CVE-2020-13630](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13630), [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-15358), [CVE-2020-13631](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13631), [CVE-2020-13632](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13632), [CVE-2020-13434](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13434), [CVE-2020-13435](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-13435), and [CVE-2020-15358](https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2020-11655).

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, avakh, baihuawei, BowenK, buxue, caifubi, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, Danish Farid, dayschan, dengwentao, dinghao, etone-chan, fangzehua, fary86, geekun, Giancarlo Colmenares, gong chen, gukecai, guohongzilong, hangangqiang, heleiwang, hesham, He Wei, hexia, hongxing, huangdongrun, huanghui, islam_amin, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, Jonathan Yan, jonyguo, Junhan Hu, Kang, kingfo, kouzhenzhong, kpy, kswang, laiyongqiang, leopz, liangzelang, lichenever, lihongkang, Li Hongzhang, lilei, limingqi107, lirongzhen1, liubuyu, liuchongming74, liuwenhao4, liuxiao, Lixia Chen, liyanliu, liyong, lizhenyu, lvliang, Mahdi, Margaret_wangrui, meixiaowei, ms_yan, nhussain, ougongchang, panfengfeng, panyifeng, peilinwang, Peilin Wang, pkuliuliu, qianlong, rick_sanchez, shibeiji, Shida He, shijianning, simson, sunsuodong, suteng, Tinazhang, Tron Zhang, unknown, VectorSL, wandongdong, wangcong, wangdongxu, wangdongxu6, wanghua, wangnan39, Wei Luning, wenchunjiang, wenkai, wilfChen, WilliamLian, wukesong, Xian Weizhao, Xiaoda Zhang, xiefangqi, xulei2020, xunxue, xutianchun, Yang, yanghaitao, yanghaitao1, yanghaoran, yangjie, yangjie159, YangLuo, Yanjun Peng, yankai, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yuchaojie, yujianfeng, zhangzhongpeng, zhangdengcheng, Zhang Qinghua, zhangyinxia, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang, wangdongxu

Contributions of any kind are welcome!

# MindSpore 0.3.1-alpha Release Notes

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- Frontend and User Interface
    - Independent model init interface.
- Data processing, augmentation, and save format
    - Support sample padding for minddataset.

## Bugfixes

- Python API
    - Fix bugs in the lars optimizer([!1894](https://gitee.com/mindspore/mindspore/pulls/1894))
- Data processing
    - Fix accuracy problem of RandomCropDecodeResize ([!2340](https://gitee.com/mindspore/mindspore/pulls/2340))

# Release 0.3.0-alpha

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- New models
    - DeepFM: a factorization-machine based neural network for CTR prediction on Criteo dataset.
    - DeepLabV3: significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2007 semantic image segmentation benchmark.
    - Faster-RCNN: towards real-time object detection with region proposal networks on COCO 2017 dataset.
    - SSD: a single stage object detection methods on COCO 2017 dataset.
    - GoogLeNet: a deep convolutional neural network architecture codenamed Inception V1 for classification and detection on CIFAR-10 dataset.
    - Wide&Deep: jointly trained wide linear models and deep neural networks for recommender systems on Criteo dataset.
- Frontend and User Interface
    - Complete numpy advanced indexing method. Supports value and assignment through tensor index.
    - Some optimizers support separating parameter groups. Different parameter groups can set different `learning_rate` and `weight_decay`.
    - Support setting submodule's logging level independently, e.g. you can set logging level of module `A` to warning and set logging level of module `B` to info.
    - Support weights to be compiled according to shape to solve the problem of large memory overhead.
    - Add some operators implement and grammar support in pynative mode. To be consistent with graph mode.
    - User interfaces change log
        - Learning rate and weight decay making group params([!637](https://gitee.com/mindspore/mindspore/pulls/637))
        - Support weights to be compiled according to shape([!1015](https://gitee.com/mindspore/mindspore/pulls/1015))
        - delete some context param([!1100](https://gitee.com/mindspore/mindspore/pulls/1100))
        - ImageSummary/ScalarSummary/TensorSummary/HistogramSummary([!1329](https://gitee.com/mindspore/mindspore/pulls/1329))([!1425](https://gitee.com/mindspore/mindspore/pulls/1425))
- Executor and Performance Optimization
    - Support doing evaluation while in training process, so that the accuracy of training can be easily obtained.
    - Enable second-order optimization for resnet50, which can achieve 75.9% accuracy in 45 epochs (Resnet50 @ImageNet).
    - Optimize pynative implementation and improve it's execution performance.
    - Optimize summary record implementation and improve its performance.
- Data processing, augmentation, and save format
    - Support simple text processing, such as tokenizer/buildvocab/lookup.
    - Support padding batch.
    - Support split or concat dataset.
    - Support MindDataset reading from file list.

### Other Hardware Support

- GPU platform
    - New models supported: MobileNetV2, MobileNetV3.
    - Support mixed precision training.
    - Support device memory swapping.

## Bugfixes

- Python API
    - An exception to the broadcast input data type check([!712](https://gitee.com/mindspore/mindspore/pulls/712))
    - Fix issues assignsub return value 0([!1036](https://gitee.com/mindspore/mindspore/pulls/1036))
    - Fix issue Conv2dBackpropInput bprop should return 3 instead of 2 items([!1001](https://gitee.com/mindspore/mindspore/pulls/1001))
    - Fix sens shape error of TrainOneStepWithLossScaleCell([!1050](https://gitee.com/mindspore/mindspore/pulls/1050))
    - Fix BatchNormGrad operator([!1344](https://gitee.com/mindspore/mindspore/pulls/1344))
- Executor
    - Fix dropout，topK and addn errors in PyNative mode ([!1285](https://gitee.com/mindspore/mindspore/pulls/1285), [!1138](https://gitee.com/mindspore/mindspore/pulls/1138), [!1033](https://gitee.com/mindspore/mindspore/pulls/1033)).
    - Fix memory leaks after execution in PyNatvie mode ([!1201](https://gitee.com/mindspore/mindspore/pulls/1201)).
    - Fix HCCL failure in some special scenes ([!1204](https://gitee.com/mindspore/mindspore/pulls/1204), [!1252](https://gitee.com/mindspore/mindspore/pulls/1252)).
    - Fix SSD network when Select failed, can't find kernel info([!1449](https://gitee.com/mindspore/mindspore/pulls/1449)).
    - Fix Topk operator selection strategy bug between aicore and aicpu([!1367](https://gitee.com/mindspore/mindspore/pulls/1367)).
    - Fix input memory size of 'assign' op unequal in control sink mode when assigning a data from one child graph to another child graph([!802](https://gitee.com/mindspore/mindspore/pulls/802)).
    - Fix allreduce ir inconsistency([!989](https://gitee.com/mindspore/mindspore/pulls/989)).
- GPU platform
    - Fix summary for gradient collection ([!1364](https://gitee.com/mindspore/mindspore/pulls/1364))
    - Fix the slice operator ([!1489](https://gitee.com/mindspore/mindspore/pulls/1489))
- Data processing
    - Fix memory problems of GeneratorDataset of sub-process ([!907](https://gitee.com/mindspore/mindspore/pulls/907))
    - Fix getting data timeout when training the cifar10 dataset under the lenet([!1391](https://gitee.com/mindspore/mindspore/pulls/1391))

## Contributors

Thanks goes to these wonderful people:

Alexey Shevlyakov, Amir Lashkari, anthony, baihuawei, biffex, buxue, caifubi, candanzg, caojian05, Cathy Wong, changzherui, chenfei, chengxianbin, chenhaozhe, chenzomi, chujinjin, cristoval, dengwentao, eric, etone-chan, fary86, gaojing, gengdongjie, gongchen, guohongzilong, guozhijian, heleiwang, hesham, He Wei, Hoai Linh Tran, hongxing, huangdongrun, huanghui, Jamie Nisbet, Jesse Lee, jiangjinsheng, jiangzhiwen, jinyaohui, jjfeing, jonwe, jonyguo, Junhan Hu, Kang, kingfo, kswang, laiyongqiang, leopz, lichenever, lihongkang, limingqi107, liubuyu, liuliyan2, liuwenhao4, liuxiao, liuxiao, liyong, lizhenyu, lvliang, Margaret_wangrui, meixiaowei, ms_yan, Nat Sutyanyong, ougongchang, panfengfeng, panyifeng, Peilin Wang, peixu_ren, qianlong, rick_sanchez, seatea, sheng, shijianning, simson, sunsuodong, Tinazhang, VectorSL, wandongdong, wangcong, wanghua, wangnan39, Wei Luning, wenchunjiang, wilfChen, WilliamLian, wsc, wukesong, wuxuejian, Xiaoda Zhang, xiefangqi, xulei2020, Yang, yangjie159, yangruoqi713, yangyongjie, yangzhenzhang, Yanjun Peng, yanzhenxiang2020, yao_yf, Yi Huaijie, yoonlee666, yujianfeng, YuJianfeng, yvetteliu, zhangdengcheng, Zhang Qinghua, zhangz0911gm, zhaojichen, zhaoting, zhaozhenlong, zhoufeng, zhouneng, zhousiyi, zhouyuanshen, Zirui Wu, Ziyan, zjun, ZPaC, lihongzhang

Contributions of any kind are welcome!

# MindSpore 0.2.0-alpha Release Notes

## Major Features and Improvements

### Ascend 910 Training and Inference Framework

- New models
    - MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    - ResNet101: Deep Residual Learning for Image Recognition.

- Frontend and User Interface
    - Support for all python comparison operators.
    - Support for math operators **,//,%. Support for other python operators like and/or/not/is/is not/ in/ not in.
    - Support for the gradients of function with variable arguments.
    - Support for tensor indexing assignment for certain indexing type.
    - Support for dynamic learning rate.
    - User interfaces change log
        - DepthwiseConv2dNative, DepthwiseConv2dNativeBackpropFilter, DepthwiseConv2dNativeBackpropInput([!424](https://gitee.com/mindspore/mindspore/pulls/424))
        - ReLU6, ReLU6Grad([!224](https://gitee.com/mindspore/mindspore/pulls/224))
        - GeneratorDataset([!183](https://gitee.com/mindspore/mindspore/pulls/183))
        - VOCDataset([!477](https://gitee.com/mindspore/mindspore/pulls/477))
        - MindDataset, PKSampler([!514](https://gitee.com/mindspore/mindspore/pulls/514))
        - map([!506](https://gitee.com/mindspore/mindspore/pulls/506))
        - Conv([!226](https://gitee.com/mindspore/mindspore/pulls/226))
        - Adam([!253](https://gitee.com/mindspore/mindspore/pulls/253))
        - _set_fusion_strategy_by_idx,_set_fusion_strategy_by_size([!189](https://gitee.com/mindspore/mindspore/pulls/189))
        - CheckpointConfig([!122](https://gitee.com/mindspore/mindspore/pulls/122))
        - Constant([!54](https://gitee.com/mindspore/mindspore/pulls/54))
- Executor and Performance Optimization
    - Support parallel execution of data prefetching and forward/backward computing.
    - Support parallel execution of gradient aggregation and forward/backward computing in distributed training scenarios.
    - Support operator fusion optimization.
    - Optimize compilation process and improve the performance.
- Data processing, augmentation, and save format
    - Support multi-process of GeneratorDataset/PyFunc for high performance
    - Support variable batchsize
    - Support new Dataset operators, such as filter,skip,take,TextLineDataset

### Other Hardware Support

- GPU platform
    - Use dynamic memory pool by default on GPU.
    - Support parallel execution of computation and communication.
    - Support continuous address allocation by memory pool.
- CPU platform
    - Support for windows 10 OS.

## Bugfixes

- Models
    - Fix mixed precision bug for VGG16 model ([!629](https://gitee.com/mindspore/mindspore/pulls/629)).
- Python API
    - Fix ControlDepend operator bugs on CPU and GPU ([!396](https://gitee.com/mindspore/mindspore/pulls/396)).
    - Fix ArgMinWithValue operator bugs ([!338](https://gitee.com/mindspore/mindspore/pulls/338)).
    - Fix Dense operator bugs on PyNative mode ([!276](https://gitee.com/mindspore/mindspore/pulls/276)).
    - Fix MatMul operator bugs on PyNative mode ([!288](https://gitee.com/mindspore/mindspore/pulls/288)).
- Executor
    - Fix operator selection bugs and make it general ([!300](https://gitee.com/mindspore/mindspore/pulls/300)).
    - Fix memory reuse bug for GetNext op ([!291](https://gitee.com/mindspore/mindspore/pulls/291)).
- GPU platform
    - Fix memory allocation in multi-graph scenarios ([!444](https://gitee.com/mindspore/mindspore/pulls/444)).
    - Fix bias_add_grad under fp16 precision ([!598](https://gitee.com/mindspore/mindspore/pulls/598)).
    - Fix support for fp16 kernels on nvidia 1080Ti([!571](https://gitee.com/mindspore/mindspore/pulls/571)).
    - Fix parsing of tuple type parameters ([!316](https://gitee.com/mindspore/mindspore/pulls/316)).
- Data processing
    - Fix TypeErrors about can't pickle mindspore._c_dataengine.DEPipeline objects([!434](https://gitee.com/mindspore/mindspore/pulls/434)).
    - Add TFRecord file verification([!406](https://gitee.com/mindspore/mindspore/pulls/406)).

## Contributors

Thanks goes to these wonderful people:

Alexey_Shevlyakov, Cathy, Chong, Hoai, Jonathan, Junhan, JunhanHu, Peilin, SanjayChan, StrawNoBerry, VectorSL, Wei, WeibiaoYu, Xiaoda, Yanjun, YuJianfeng, ZPaC, Zhang, ZhangQinghua, ZiruiWu, amongo, anthonyaje, anzhengqi, biffex, caifubi, candanzg, caojian05, casgj, cathwong, ch-l, chang, changzherui, chenfei, chengang, chenhaozhe, chenjianping, chentingting, chenzomi, chujinjin, dengwentao, dinghao, fanglei, fary86, flywind, gaojing, geekun, gengdongjie, ghzl, gong, gongchen, gukecai, guohongzilong, guozhijian, gziyan, h.farahat, hesham, huangdongrun, huanghui, jiangzhiwen, jinyaohui, jjfeing, jojobugfree, jonathan_yan, jonyguo, jzw, kingfo, kisnwang, laiyongqiang, leonwanghui, lianliguang, lichen, lichenever, limingqi107, liubuyu, liuxiao, liyong, liyong126, lizhenyu, lupengcheng, lvliang, maoweiyong, ms_yan, mxm, ougongchang, panfengfeng, panyifeng, pengyanjun, penn, qianlong, seatea, simson, suteng, thlinh, vlne-v1, wangchengke, wanghua, wangnan39, wangqiuliang, wenchunjiang, wenkai, wukesong, xiefangqi, xulei, yanghaitao, yanghaoran, yangjie159, yangzhenzhang, yankai10, yanzhenxiang2020, yao_yf, yoonlee666, zhangbuxue, zhangz0911gm, zhangzheng, zhaojichen, zhaoting, zhaozhenlong, zhongligeng, zhoufeng, zhousiyi, zjun, zyli2020, yuhuijun, limingqi107, lizhenyu, chenweifeng.

Contributions of any kind are welcome!

# MindSpore 0.1.0-alpha Release Notes

## Main Features

### Ascend 910 Training and Inference Framework

- Recommended OS: Ubuntu 16.04 (or later) or EulerOS 2.5 or EulerOS 2.8
- Python version: 3.7.5
- Preset models
    - ResNet-50: residual structure-based convolutional neural network (CNN) for image classification, which is widely used.
    - AlexNet: classic CNN for image classification, achieving historical results in ImageNet LSVRC-2012.
    - LeNet: classic CNN for image classification, which was proposed by Yann LeCun.
    - VGG16: classic CNN for image classification, which was proposed by Oxford Visual Geometry Group.
    - YoloV3: real-time object detection network.
    - NEZHA: BERT-based Chinese pre-training network produced by Huawei Noah's Ark Laboratory.
- Execution modes
    - Graph mode: provides graph optimization methods such as memory overcommitment, IR fusion, and buffer fusion to achieve optimal execution performance.
    - PyNative mode: single-step execution mode, facilitating process debugging.
- Debugging capability and methods
    - Save CheckPoints and Summary data during training.
    - Support asynchronous printing.
    - Dump the computing data.
    - Support profiling analysis of the execution process performance.
- Distributed execution
    - Support AllReduce, AllGather, and BroadCast collective communication.
    - AllReduce data parallel: Each device obtains different training data, which accelerates the overall training process.
    - Collective communication-based layerwise parallel: Models are divided and allocated to different devices to solve the problem of insufficient memory for large model processing and improve the training speed.
    - Automatic parallel mode: The better data and model parallel mode can be predicted based on the cost model. It is recommended that this mode be used on ResNet series networks.
- Automatic differentiation
    - Implement automatic differentiation based on Source to Source.
    - Support distributed scenarios and automatic insertion of reverse communication operators.
- Data processing, augmentation, and save format
    - Load common datasets such as ImageNet, MNIST, CIFAR-10, and CIFAR-100.
    - Support common data loading pipeline operations, such as shuffle, repeat, batch, map, and sampler.
    - Provide basic operator libraries to cover common CV scenarios.
    - Support users to customize Python data augmentation operators through the Pyfunc mechanism.
    - Support the access of user-defined datasets through the GeneratorDataset mechanism.
    - Provide the MindSpore data format, data aggregation and storage, random access example, data partition, efficient parallel read, user-defined index, and dataset search.
    - Convert user datasets to the MindSpore data format.
    - After data processing and augmentation, provide training applications in feed and graph modes.
- FP32/16 mixed precision computation, supporting automatic and manual configuration
- Provide common operators such as nn, math, and array, which can be customized.

### Inference Deployment

- Deploy models in MindSpore format on the Ascend 310 platform for inference.
- Save models in ONNX format.
- Support saving models in LITE format and running models based on the lightweight inference framework.
    - Recommended OS: Android 4.3 or later
    - Supported network type: LeNet
    - Provide the generalization operators generated by TVM and operators generated after specific networks are tuned.

### Other Hardware Support

- GPU platform training
    - Recommended OS: Ubuntu 16.04
    - CUDA version: 9.2 or 10.1
    - CuDNN version: 7.6 or later
    - Python version: 3.7.5
    - NCCL version: 2.4.8-1
    - OpenMPI version: 3.1.5
    - Supported models: AlexNet, LeNet, and LSTM
    - Supported datasets: MNIST and CIFAR-10
    - Support data parallel.
- CPU platform training
    - Recommended OS: Ubuntu 16.04
    - Python version: 3.7.5
    - Supported model: LeNet
    - Supported dataset: MNIST
    - Provide only the stand-alone operation version.

## Peripherals and Tools

- [MindSpore Official Website](https://www.mindspore.cn/)
- [MindInsight Visualization Debugging and Optimization](https://gitee.com/mindspore/mindinsight)
- [MindArmour Model Security Hardening Package](https://gitee.com/mindspore/mindarmour)
- [GraphEngine Computational Graph Engine](https://gitee.com/mindspore/graphengine)
