/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_ANF_RUNTIME_ALGORITHM_H
#define MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_ANF_RUNTIME_ALGORITHM_H
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <memory>
#include <map>
#include <optional>
#include "ir/anf.h"
#include "base/base.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/runtime/hardware_abstract/kernel_base/oplib/opinfo.h"
#include "include/utils/contract.h"
#include "device_address/device_address.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace device {
class KernelInfo;
}
namespace session {
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;
using Address = kernel::Address;
using AddressPtr = kernel::AddressPtr;
using kernel::KernelObjectType;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using AnfWithOutIndex = std::pair<AnfNodePtr, size_t>;
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
class KernelGraph;
using KernelGraphPtr = std::shared_ptr<KernelGraph>;

class BACKEND_COMMON_EXPORT AnfRuntimeAlgorithm {
 public:
  // Get the memory size of output tensor of node.
  static size_t GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index);
  static size_t GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index, const ShapeVector &shape);
  // get all outputs format select of anf node
  static std::vector<std::string> GetAllOutputFormats(const AnfNodePtr &node);
  // get all inputs format select of anf node
  static std::vector<std::string> GetAllInputFormats(const AnfNodePtr &node);
  // get all inputs type select of anf node
  static std::vector<TypeId> GetAllInputDeviceTypes(const AnfNodePtr &node);
  // get all outputs type select of anf node
  static std::vector<TypeId> GetAllOutputDeviceTypes(const AnfNodePtr &node);
  // get origin data format select of anf node
  static std::string GetOriginDataFormat(const AnfNodePtr &node);
  // get output format select of anf node
  static std::string GetOutputFormat(const AnfNodePtr &node, size_t output_idx);
  // get input format select of anf node
  static std::string GetInputFormat(const AnfNodePtr &node, size_t input_idx);
  // Judge whether the format is equivalent by converting between default format and real format.
  static bool IsEquivalentFormat(const Format &src_format, const Format &dst_format);
  // get output format from prev node,input_index is the input index of current node related to prev node
  static std::string GetPrevNodeOutputFormat(const AnfNodePtr &anf_node, size_t input_idx);
  // get reshape_type of from the output of input node.
  static std::string GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx);
  // get output shapes which will built and run in device
  static std::vector<int64_t> GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx);
  // get output shapes which will built and run in device when dynamic shape
  static std::vector<int64_t> GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx, ShapeVector real_shape);
  // get input shapes which will built and run in device
  static std::vector<int64_t> GetInputDeviceShape(const AnfNodePtr &node, size_t input_idx);
  // get input kernel object type
  static std::vector<KernelObjectType> GetInputKernelObjectTypes(const AnfNodePtr &node);
  static KernelObjectType GetInputKernelObjectType(const AnfNodePtr &node, size_t input_idx);
  // get output kernel object type
  static std::vector<KernelObjectType> GetOutputKernelObjectTypes(const AnfNodePtr &node);
  static KernelObjectType GetOutputKernelObjectType(const AnfNodePtr &node, size_t output_idx);
  // get output kernel object type
  static std::vector<KernelObjectType> GetOutputElementsKernelObjectTypes(const AnfNodePtr &node);
  // Get Input Padding Axis
  static std::string GetInputReshapeType(const AnfNodePtr &node, size_t input_idx);
  // Get Output Padding Axis
  static std::string GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx);
  // Get all input reshape shape of anf node
  static std::vector<std::string> GetAllInputReshapeType(const AnfNodePtr &node);
  // Get all output reshape shape of anf node
  static std::vector<std::string> GetAllOutputReshapeType(const AnfNodePtr &node);
  // get output select data type of anf node
  static TypeId GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx);
  // get input select data type of anf node
  static TypeId GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx);
  // get output select data type from prev node,input_index is the input index of current node related to prev node
  static TypeId GetPrevNodeOutputDeviceDataType(const AnfNodePtr &anf_node, size_t input_idx);
  // get output device addr of anf_node
  static const DeviceAddress *GetOutputAddr(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node = true);
  // get mutable output device addr of anf_node
  static DeviceAddressPtr GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node = true);
  static DeviceAddressPtr GetMutableOutputAddr(const KernelWithIndex &node_output_index, bool skip_nop_node) {
    return GetMutableOutputAddr(node_output_index.first, node_output_index.second, skip_nop_node);
  }
  // get storage_info
  static TensorStorageInfoPtr GetMutableOutputStorageInfo(const AnfNodePtr &node, size_t output_idx,
                                                          bool skip_nop_node = true);
  static TensorStorageInfoPtr GetMutableOutputStorageInfo(const KernelWithIndex &node_output_index,
                                                          bool skip_nop_node) {
    return GetMutableOutputStorageInfo(node_output_index.first, node_output_index.second, skip_nop_node);
  }
  // check whether output addr is exist or not
  static bool OutputAddrExist(const AnfNodePtr &node, size_t output_idx, bool skip_nop_node = false);
  // check whether workspace addr is exist or not
  static bool WorkspaceAddrExist(const AnfNodePtr &node, size_t output_idx);
  // get address from prev node,input_index is the input index of current node related to prev node
  static const DeviceAddress *GetPrevNodeOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                    bool skip_nop_node = true);
  static DeviceAddressPtr GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                       bool skip_nop_node = true);
  // get storage_info from prev node,input_index is the input index of current node related to prev node
  static TensorStorageInfoPtr GetPrevNodeMutableOutputStorageInfo(const AnfNodePtr &anf_node, size_t input_idx,
                                                                  bool skip_nop_node = true);
  // Get shape, devie type and value information.
  static std::tuple<abstract::BaseShapePtr, TypePtr, ValuePtr> GetAbstractInfo(const AnfNodePtr &node,
                                                                               size_t output_idx);
  static bool ExistOutputKernelTensor(const AnfNodePtr &node, size_t output_idx);

  // Get output kernel tensor if exists, otherwise throw a exception.
  static const KernelTensorPtr &GetOutputKernelTensor(const AnfNodePtr &node, size_t output_idx,
                                                      bool skip_nop_node = true);
  // Get output kernel tensor if exists, otherwise create a new one and set into node.
  static const KernelTensorPtr &GetOrCreateOutputKernelTensor(const AnfNodePtr &node, size_t output_idx);

  // Get input kernel tensor if exists, otherwise throw a exception.
  static const KernelTensorPtr &GetPrevNodeOutputKernelTensor(const AnfNodePtr &node, size_t input_idx,
                                                              bool skip_nop_node = true);
  // Get input kernel tensor if exists, otherwise create a new one and set into node.
  static const KernelTensorPtr &GetOrCreatePrevNodeOutputKernelTensor(const AnfNodePtr &node, size_t input_idx);

  // Get all input kernel tensor if exists, otherwise create new KernelTensor and set into input node.
  static std::vector<KernelTensor *> GetOrCreateAllInputKernelTensors(const AnfNodePtr &node);
  // Get all output kernel tensor if exists, otherwise create new KernelTensor and set into node.
  static std::vector<KernelTensor *> GetOrCreateAllOutputKernelTensors(const AnfNodePtr &node);

  // Set kernel object type
  static void SetKernelObjectTypeBuildInfo(const AnfNodePtr &kernel_node,
                                           const std::vector<kernel::KernelObjectType> &input_kernel_object_types,
                                           const std::vector<kernel::KernelObjectType> &output_kernel_object_types);
  static void SetKernelObjectTypeBuildInfo(
    const AnfNodePtr &kernel_node, const std::vector<kernel::KernelObjectType> &input_kernel_object_types,
    const std::vector<kernel::KernelObjectType> &output_kernel_object_types,
    const std::vector<kernel::KernelObjectType> &output_elements_kernel_object_types);
  static void SetKernelObjectTypeWithSelectedAttr(const CNodePtr &kernel_node,
                                                  const kernel::KernelAttr &selected_kernel_attr);
  static bool SelectKernelByObjectType(const CNodePtr &kernel_node,
                                       const std::vector<kernel::KernelAttr> &registered_kernel_attrs,
                                       std::vector<kernel::KernelAttr> *selected_kernel_attrs);
  // Create output kernel tensor for node using node's shape, type and value,
  // and set device information to kernel tensor.
  static KernelTensorPtr CreateOutputKernelTensorWithDeviceInfo(
    const AnfWithOutIndex &node_with_index, void *const device_ptr, size_t size, const string &format, TypeId dtype_id,
    const ShapeVector &host_shape, const std::string &device_name, uint32_t device_id,
    const UserDataPtr &user_data = nullptr, uint32_t stream_id = 0, bool is_remote = false);

  // Get all input memory size list for node.
  static std::vector<size_t> GetNodeInputSizeList(const AnfNodePtr &node);

  static size_t GetOutputAddressNum(const AnfNodePtr &node);
  // set output device addr of anf_node
  static void SetOutputAddr(const DeviceAddressPtr &addr, const TensorStorageInfoPtr &storage_info, size_t output_idx,
                            const AnfNodePtr &node);
  // set output kernel tensor of anf node
  static void SetOutputKernelTensor(const KernelTensorPtr &kernel_tensor, size_t output_idx, AnfNode *node);
  // set alloc stream id
  static void SetAllocStreamId(const KernelTensorPtr &kernel_tensor, uint32_t stream_id, const AnfNodePtr &node);
  // set workspace device addr of anf_node
  static void SetWorkspaceAddr(const DeviceAddressPtr &addr, const TensorStorageInfoPtr &storage_info,
                               size_t output_idx, const AnfNodePtr &node);
  // set workspace kernel tensor of anf_node
  static void SetWorkspaceKernelTensor(const KernelTensorPtr &kernel_tensor, size_t output_idx, AnfNode *node);
  // get workspace device addr of anf_node
  static DeviceAddress *GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx);
  static KernelTensorPtr GetWorkspaceKernelTensor(const AnfNodePtr &node, size_t output_idx);
  // get workspace device mutable addr of anf_node
  static DeviceAddressPtr GetMutableWorkspaceAddr(const AnfNodePtr &node, size_t index);
  // get op pattern of the node
  static kernel::OpPattern GetOpPattern(const AnfNodePtr &node);
  // get KernelBuildType of node ,such as ATT,RT,FWK and so on
  static KernelType GetKernelType(const AnfNodePtr &node);
  // get processor type:AICORE,AICPU...
  static kernel::Processor GetProcessor(const AnfNodePtr &node);
  // get fusion type:AICORE,AICPU...
  static std::string GetFusionType(const AnfNodePtr &node);
  static void SetFusionType(const AnfNodePtr &node, const std::string &type);
  // get KernelBuildInfoValid
  static bool GetValid(const AnfNodePtr &node);

  static void SetOutputDataDesc(const AnfNodePtr &node, const std::vector<nlohmann::json> &desc);
  static std::vector<nlohmann::json> GetOutputDataDesc(const AnfNodePtr &node);
  // core type
  static void SetCoreType(const AnfNodePtr &node, const std::string &core_type);
  static std::string GetCoreType(const AnfNodePtr &node);
  // op type
  static kernel::OpType GetOpType(const AnfNodePtr &node);
  // set select kernel_build_info
  static void SetSelectKernelBuildInfo(const kernel::KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node);
  // get select kernel_build_info
  static kernel::KernelBuildInfoPtr GetSelectKernelBuildInfo(const AnfNodePtr &node);
  static kernel::KernelAttr GetKernelAttrFromNode(const AnfNodePtr &kernel_node);
  // get kernelMode
  static kernel::KernelMod *GetKernelMod(const AnfNodePtr &node);
  // set kernel mod
  static void SetKernelMod(const kernel::KernelModPtr &kernel_mod, AnfNode *node);
  // set stream id of kernel,which will be set in stream assign and be used in stream generate
  static void SetStreamId(uint32_t stream_id, AnfNode *node);
  // get stream id
  static uint32_t GetStreamId(const AnfNodePtr &node);
  // set stream distinction label to distinguish different ops in different streams
  static void SetStreamDistinctionLabel(uint32_t stream_label, AnfNode *node);
  // get stream distinction label
  static uint32_t GetStreamDistinctionLabel(const AnfNode *node);
  // set graph id
  static void SetGraphId(uint32_t graph_id, AnfNode *node);
  // get graph id
  static uint32_t GetGraphId(const AnfNode *node);
  static std::vector<KernelGraphPtr> GetCallSwitchKernelGraph(const CNodePtr &cnode);
  static KernelGraphPtr GetValueNodeKernelGraph(const AnfNodePtr &node);
  static KernelGraphPtr FetchKernelGraph(const AnfNode *node);
  static AnfNodePtr FetchFrontNodeByBackendNode(const AnfNodePtr &backend_node, const KernelGraph &graph);
  static void InsertMakeTupleForOutput(const NotNull<KernelGraphPtr> &root_graph);

  // get jit settings from kernel graph
  static std::string GetJitLevel(const FuncGraphPtr &graph);
  static std::string GetBackend(const FuncGraphPtr &graph);
  static bool GetDisableFormatTransform(const KernelGraphPtr &graph);
  static std::string GetExecOrderAlgo(const KernelGraphPtr &graph);
  static std::map<std::string, std::map<std::string, std::string>> GetGeOptions(const KernelGraphPtr &graph);
  // get ge options from jitconfig or context
  static std::map<std::string, std::string> GetGeOptions(std::string option_level);

  static void UpdateGraphValidRefPair(const KernelGraphPtr &graph);
  static bool IsDynamicShapeSkipExecute(bool skip_mode, const ShapeVector &axes_shape);
  static bool IsShapesDynamic(const std::vector<ShapeVector> &shapes);

  // Get shape after padding
  static ShapeVector GetRuntimePaddingShape(const AnfNodePtr &node, size_t index);

  static void AddOutInRefToGraph(const KernelGraphPtr &graph);

  static bool NodeValueIsFuncGraph(const AnfNodePtr &node);

  // Whether the kernel is not supported by other device and need be backed off on the CPU device.
  static bool IsNodeSupportKernelSelectBackoff(const AnfNodePtr &node, const KernelGraphPtr &graph);
  static bool IsKernelSelectBackoffOp(const AnfNodePtr &node);
  static bool IsNeedContinuesMemoryOp(const AnfNodePtr &node);
  static void SetKernelSelectBackoffInfo(const CNodePtr &node,
                                         const std::pair<std::string, ExceptionType> &failure_info);
  static std::pair<std::string, ExceptionType> GetKernelSelectBackoffInfo(const AnfNodePtr &node);

  // The related interface of device target.
  static std::string FetchDeviceTarget(const AnfNodePtr &node, const KernelGraph *graph);
  // Set device target for parameter affinity by the user nodes in the graph.
  static void SetParameterDeviceTarget(const KernelGraphPtr graph);

  // Get the real output num(which can be build and run in device).
  static size_t GetOutputTensorNum(const AnfNodePtr &node);
  // Get the expanded output element num(which the tuple is expanded to calculate num).
  static size_t GetOutputElementNum(const AnfNodePtr &node);

  // Get output abstract type of anf node.
  static TypeId GetAbstractObjectType(const AbstractBasePtr &abstract);
  static TypeId GetOutputObjectType(const AnfNodePtr &node, size_t output_idx);
  static TypeId GetInputObjectType(const CNodePtr &node, size_t input_idx);
  static std::vector<TypeId> GetAllInputObjectType(const AnfNodePtr &node);
  static std::vector<TypeId> GetAllOutputObjectType(const AnfNodePtr &node);
  // Get unfold input num
  static size_t GetInputElementNum(const AnfNodePtr &node);
  static bool IsRealSquenceOutput(const AnfNodePtr &node);
  static void SetDynamicAttrToPrim(const PrimitivePtr &prim);

  // Get output detail shape. These interfaces should take TUPLE output into consideration.
  static abstract::BaseShapePtr GetOutputDetailShape(const AnfNodePtr &node, size_t output_idx);
  static abstract::BaseShapePtr GetPrevNodeOutputDetailShape(const AnfNodePtr &node, size_t input_idx);

  // Check whether the input scalar need converted to tensor.
  static bool IsScalarConvertToTensor(const AnfNodePtr &input_node, const CNodePtr &node);
  // Check all elements of a ndoe's output(tuple/list type) are scalar.
  static bool IsSequenceOutputOfScalar(const AnfNodePtr &node);

  static void FlattenDynamicInputArg(const BaseRef &arg, const AnfNodePtr &node,
                                     std::vector<tensor::TensorPtr> *flatten_tensors);
  static void FlattenInputArg(const BaseRef &arg, const AnfNodePtr &node,
                              std::vector<tensor::TensorPtr> *flatten_tensors);

  static void UpdateValueNodeShape(const AnfNodePtr &node);
  static bool HasSelectKernelBuildInfo(const AnfNodePtr &node);
  static bool NeedEraseCache(const PrimitivePtr &prim);

  static abstract::AbstractBasePtr GetNodeAbstractByIndex(const AnfNodePtr &node, size_t index);
  static abstract::AbstractBasePtr GetNodeAbstractByIndex(AnfNode *node, size_t index);

  static ValueNodePtr ConvertValueToNode(const KernelGraphPtr &kernel_graph, const ValuePtr &value);
  // create type id value node and add it to graph
  static ValueNodePtr CreateTypeIdValueNodeToKernelGraph(const FuncGraphPtr &func_graph, TypeId data_type);
  static ValueNodePtr CreateTypeIdValueNodeToFuncGraph(const FuncGraphPtr &func_graph, TypeId data_type);
  static bool IsNoRealKernelGraph(const KernelGraphPtr &kernel_graph);

  // if graph output is valuenode or parameter, used to skip compile
  static bool IsGraphOutputValueNodeOrParameterForCompile(const AnfNodePtr &graph_output);

  // Only used for ascend ops.
  static bool IsLaunchIgnoredInputAddressIdx(const AnfNodePtr &node, size_t input_idx);
  static std::string GetValueByDeviceAddress(KernelTensor *const device_address, size_t element_num);
  static void PrintKernelTensor(const std::vector<KernelTensor *> &kernel_tensors, const std::string &info,
                                size_t element_num);
  static KernelTensorPtr CreateKernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type,
                                            const ValuePtr &value, void *device_ptr, size_t size,
                                            const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
                                            const string &device_name, uint32_t device_id,
                                            const UserDataPtr &user_data = nullptr, bool is_remote = false);
  static KernelTensorPtr CreateKernelTensor(void *device_ptr, size_t size, Format format, TypeId dtype_id,
                                            const ShapeVector &host_shape, const string &device_name,
                                            uint32_t device_id, const UserDataPtr &user_data = nullptr,
                                            bool is_remote = false);

  // Get device attr string from Parameter.
  static std::string GetParameterDeviceStr(const mindspore::AnfNodePtr &node);
  // check if is GE backend
  static bool IsBackendGe();
  // check if is ms_backend backend
  static bool IsBackendMs();
  static bool ParseMetadata(const CNodePtr &kernel_node, const std::shared_ptr<const kernel::OpInfo> &op_info_ptr,
                            kernel::Processor processor,
                            std::vector<std::shared_ptr<kernel::KernelBuildInfo>> *const kernel_info_list);
  static void SetDynamicInputSizeAttr(const CNodePtr &cnode);
  static int64_t CalOutputTupleSize(const AnfNodePtr &node);
  static void UnfoldKernelBuildInfo(const CNodePtr &kernel_node);
  static std::pair<std::string, ExceptionType> KernelObjectTypeNotSupportWarning(const CNodePtr &kernel_node);
};
}  // namespace session

using AnfAlgo = session::AnfRuntimeAlgorithm;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_BACKEND_COMMON_KERNEL_GRAPH_ANF_RUNTIME_ALGORITHM_H
