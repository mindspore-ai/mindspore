/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_ANF_RUNTIME_ALGORITHM_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_ANF_RUNTIME_ALGORITHM_H
#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <memory>
#include <unordered_set>
#include <map>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "ir/kernel_info_dev.h"
#include "runtime/device/device_address.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/kernel_compiler/kernel_build_info.h"
#include "base/core_ops.h"
#include "utils/contract.h"
#include "backend/session/kernel_graph.h"

namespace mindspore {
namespace session {
using PrimitiveSet = std::unordered_set<PrimitivePtr, PrimitiveHasher, PrimitiveEqual>;
using AnfVisitFuncion = std::function<Any(const AnfNodePtr &node, int index)>;
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;
using Address = kernel::Address;
using AddressPtr = kernel::AddressPtr;

class OpRuntimeInfo {
 public:
  OpRuntimeInfo(std::vector<std::string> output_format, std::vector<TypeId> output_type,
                std::vector<size_t> output_tensor_size)
      : output_format_(std::move(output_format)),
        output_type_(std::move(output_type)),
        output_tensor_size_(std::move(output_tensor_size)) {}
  ~OpRuntimeInfo() = default;

  // Key for user data.
  constexpr static char key[] = "OpRuntimeInfo";

  std::string output_format(size_t index) const;
  TypeId output_type(size_t index) const;
  size_t output_tensor_size(size_t index) const;

 private:
  std::vector<std::string> output_format_;
  std::vector<TypeId> output_type_;
  std::vector<size_t> output_tensor_size_;
};

class AnfRuntimeAlgorithm {
 public:
  static AnfNodePtr MakeMonadValueNode(const KernelGraphPtr &kg);
  static void KeepOrder(const KernelGraphPtr &kg, const AnfNodePtr &former, const AnfNodePtr &latter);
  // get real input node of tuple_get_item
  static AnfNodePtr GetTupleGetItemRealInput(const CNodePtr &tuple_get_item);
  static size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item);
  // get input_anf_node's real kernel by recurse
  static KernelWithIndex VisitKernel(const AnfNodePtr &input_anf_node, size_t output_index);
  static KernelWithIndex VisitKernelWithReturnType(const AnfNodePtr &input_anf_node, size_t output_index,
                                                   bool skip_nop_node = false,
                                                   const std::vector<PrimitivePtr> &return_types = {
                                                     prim::kPrimMakeTuple});
  static std::vector<AnfNodePtr> GetAllOutput(const AnfNodePtr &node,
                                              const std::vector<PrimitivePtr> &return_types = {});
  static std::vector<KernelWithIndex> GetAllOutputWithIndex(const AnfNodePtr &node);
  // get cnode primitive
  static AnfNodePtr GetCNodePrimitiveNode(const CNodePtr &node);
  static void SetNodeInput(const CNodePtr &node, const AnfNodePtr &input_node, size_t index);
  static PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node);
  // check whether anf node is a node of 'primitive_type',such as make_tuple is a cnode of kPrimMakeTuple
  static bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type);
  // get cnode primitive
  static FuncGraphPtr GetCNodeFuncGraphPtr(const AnfNodePtr &node);
  // get kernel_name of anf node
  static std::string GetCNodeName(const AnfNodePtr &node);
  // get detail info of anf node
  static std::string GetNodeDebugString(const AnfNodePtr &node);
  // get attr of anf node
  template <typename T>
  static T GetNodeAttr(const AnfNodePtr &node, const std::string &key) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      std::string node_debug_log = node->DebugString();
      MS_LOG(EXCEPTION) << "Only cnode has attr, but this anf is " << node_debug_log.c_str();
    }
    // single op cnode.
    if (auto primitive = GetCNodePrimitive(node); primitive != nullptr) {
      return GetValue<T>(primitive->GetAttr(key));
    }
    // graph kernel cnode.
    auto fg = GetCNodeFuncGraphPtr(node);
    MS_EXCEPTION_IF_NULL(fg);
    return GetValue<T>(fg->get_attr(key));
  }
  static bool IsTupleOutput(const AnfNodePtr &anf);
  // set attr of anf node
  static void SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node);
  // set attr of key from 'from' node to 'to' node
  static void CopyNodeAttr(const std::string &key, const AnfNodePtr &from, const AnfNodePtr &to);
  // set a new key for attr from 'from' node to 'to' node
  static void CopyNodeAttr(const std::string &old_key, const std::string &new_key, const AnfNodePtr &from,
                           const AnfNodePtr &to);
  // set all attrs from 'from' node to 'to' node
  static void CopyNodeAttrs(const AnfNodePtr &from, const AnfNodePtr &to);
  // check whether a cnode has the specified attr.
  static bool HasNodeAttr(const std::string &key, const CNodePtr &node);
  // delete attr of anf node
  static void EraseNodeAttr(const std::string &key, AnfNodePtr node);
  // get the num of inputs include monads for a cnode
  static size_t GetInputNum(const CNodePtr &cnode);
  // get the num of inputs exclude monads for real_kernel (which can be build and run in device)
  static size_t GetInputTensorNum(const AnfNodePtr &node);
  // get the num of output real_kernel(which can be build and run in device)
  static size_t GetOutputTensorNum(const AnfNodePtr &node);
  // Get the memory size of output tensor of node.
  static size_t GetOutputTensorMemSize(const AnfNodePtr &node, size_t output_index);
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
  // get prev node output width output index
  static KernelWithIndex GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx, bool visit_nop_node = false);
  // get output format from prev node,input_index is the input index of current node related to prev node
  static std::string GetPrevNodeOutputFormat(const AnfNodePtr &node, size_t input_idx);
  // get reshape_type of from the output of input node.
  static std::string GetPrevNodeOutputReshapeType(const AnfNodePtr &node, size_t input_idx);
  // get output shapes inferred by ME from input nodes.
  static std::vector<size_t> GetOutputInferShape(const AnfNodePtr &node, size_t output_idx);
  static std::vector<size_t> GetOutputInferShape(const AnfNodePtr &node, const abstract::BaseShapePtr &base_shape,
                                                 size_t output_idx);
  // get input shapes inferred by ME from input nodes.
  static std::vector<size_t> GetPrevNodeOutputInferShape(const AnfNodePtr &node, size_t input_idx);
  // get output shapes which will built and run in device
  static std::vector<size_t> GetOutputDeviceShape(const AnfNodePtr &node, size_t output_idx);
  // get input shapes which will built and run in device
  static std::vector<size_t> GetInputDeviceShape(const AnfNodePtr &node, size_t input_idx);
  // get output shapes for tbe build
  static std::vector<int64_t> GetOutputDeviceShapeForTbeBuild(const AnfNodePtr &node, const size_t output_idx,
                                                              const std::string &format);
  // Get Input Padding Axis
  static std::string GetInputReshapeType(const AnfNodePtr &node, size_t output_idx);
  // Get Output Padding Axis
  static std::string GetOutputReshapeType(const AnfNodePtr &node, size_t output_idx);
  // get output data type inferred by ME of anf node
  static TypeId GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx);
  static TypeId GetOutputInferDataType(const TypePtr &type_ptr, size_t output_idx);
  // get output original data type from prev node,input_index is the input index of current node related to prev node
  static TypeId GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx);
  // get output select data type of anf node
  static TypeId GetOutputDeviceDataType(const AnfNodePtr &node, size_t output_idx);
  // get input select data type of anf node
  static TypeId GetInputDeviceDataType(const AnfNodePtr &node, size_t input_idx);
  // get output select data type from prev node,input_index is the input index of current node related to prev node
  static TypeId GetPrevNodeOutputDeviceDataType(const AnfNodePtr &node, size_t input_idx);
  // get output device addr of anf_node
  static const DeviceAddress *GetOutputAddr(const AnfNodePtr &node, size_t output_idx, bool visit_nop_node = true);
  // get mutable output device addr of anf_node
  static DeviceAddressPtr GetMutableOutputAddr(const AnfNodePtr &node, size_t output_idx, bool visit_nop_node = true);
  // check whether output addr is exist or not
  static bool OutputAddrExist(const AnfNodePtr &node, size_t output_idx, bool visit_nop_node = false);
  // check whether workspace addr is exist or not
  static bool WorkspaceAddrExist(const AnfNodePtr &node, size_t output_idx);
  // get address from prev node,input_index is the input index of current node related to prev node
  static const DeviceAddress *GetPrevNodeOutputAddr(const AnfNodePtr &node, size_t input_idx,
                                                    bool visit_nop_node = true);
  static DeviceAddressPtr GetPrevNodeMutableOutputAddr(const AnfNodePtr &anf_node, size_t input_idx,
                                                       bool visit_nop_node = true);
  // set output device addr of anf_node
  static void SetOutputAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node);
  // set workspace device addr of anf_node
  static void SetWorkspaceAddr(const DeviceAddressPtr &addr, size_t output_idx, AnfNode *node);
  // get workspace device addr of anf_node
  static DeviceAddress *GetWorkspaceAddr(const AnfNodePtr &node, size_t output_idx);
  // get workspace device mutable addr of anf_node
  static DeviceAddressPtr GetMutableWorkspaceAddr(const AnfNodePtr &node, size_t index);
  // set infer shapes and types of anf node
  static void SetOutputInferTypeAndShape(const std::vector<TypeId> &types,
                                         const std::vector<std::vector<size_t>> &shapes, AnfNode *node);
  // get and set output shape ptr
  static abstract::BaseShapePtr GetOutputDetailShape(const AnfNodePtr &node, size_t output_idx);
  static abstract::BaseShapePtr GetPrevNodeOutputDetailShape(const AnfNodePtr &node, size_t input_idx);
  static void SetOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                          const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node);
  static void CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node);
  // get op pattern of the node
  static kernel::OpPattern GetOpPattern(const AnfNodePtr &node);
  // get KernelBuildType of node ,such as ATT,RT,FWK and so on
  static KernelType GetKernelType(const AnfNodePtr &node);
  // get processor type:AICORE,AICPU...
  static kernel::Processor GetProcessor(const AnfNodePtr &node);
  // get fusion type:AICORE,AICPU...
  static kernel::FusionType GetFusionType(const AnfNodePtr &node);
  static void SetFusionType(const AnfNodePtr &node, const kernel::FusionType &type);
  static void SetOutputDataDesc(const AnfNodePtr &node, const std::vector<nlohmann::json> &desc);
  static std::vector<nlohmann::json> GetOutputDataDesc(const AnfNodePtr &node);
  // set select kernel_build_info
  static void SetSelectKernelBuildInfo(const kernel::KernelBuildInfoPtr &select_kernel_build_info, AnfNode *node);
  // get select kernel_build_info
  static kernel::KernelBuildInfoPtr GetSelectKernelBuildInfo(const AnfNodePtr &node);
  // get kernelMode
  static kernel::KernelMod *GetKernelMod(const AnfNodePtr &node);
  // set kernel mod
  static void SetKernelMod(const kernel::KernelModPtr &kernel_mod, AnfNode *node);
  // checkout whether the anf node is a real kernel that can run on device,parameter and constant is real kernel too
  static bool IsRealKernel(const AnfNodePtr &node);
  // checkout whether the anf node is a real kernel that is a cnode and can run on device
  static bool IsRealCNodeKernel(const AnfNodePtr &node);
  // checkout whether the anf node is a graph kernel.
  static bool IsGraphKernel(const AnfNodePtr &node);
  // checkout whether the anf node is an inner node of graph kernel.
  static bool IsNodeInGraphKernel(const AnfNodePtr &node);
  // get the real output of GraphKernel.
  static AnfNodePtr GetOutputOfGraphkernel(const KernelWithIndex &kernel_with_index);
  // check parameter is weight or data
  static bool IsParameterWeight(const ParameterPtr &node);
  // checkout whether the anf node is include the label_index.
  static bool IsLabelIndexInNode(const AnfNodePtr &node, size_t label_index);
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
  static AnfNodePtr GetInputNode(const CNodePtr &node, size_t index);
  // charge if the node's output is a feature map output
  static bool IsFeatureMapOutput(const AnfNodePtr &node);
  // charge if the node's input is from a feature map output
  static bool IsFeatureMapInput(const AnfNodePtr &node, size_t input_index);
  // get real input index for some tbe ops which input order is different between me and tbe impl
  static size_t GetRealInputIndex(const AnfNodePtr &anf_node, const size_t cur_index);
  // get me input index for some tbe ops which input order is different between me and tbe impl
  static size_t GetOriginalInputIndex(const AnfNodePtr &anf_node, const size_t cur_index);
  static bool IsCommunicationOp(const AnfNodePtr &node);
  static bool IsFusedCommunicationOp(const AnfNodePtr &node);
  static bool IsInplaceNode(const AnfNodePtr &node, const string &type);
  static bool IsGetNext(const NotNull<AnfNodePtr> &node);
  static FuncGraphPtr GetValueNodeFuncGraph(const AnfNodePtr &node);
  static std::vector<KernelGraphPtr> GetCallSwitchKernelGraph(const CNodePtr &cnode);
  static bool IsSwitchCall(const CNodePtr &call_node);
  static bool IsScalarInput(const CNodePtr &cnode, size_t index);
  static bool IsScalarOutput(const CNodePtr &cnode, size_t index);
  static void ReorderExecList(NotNull<std::vector<CNodePtr> *> node_list);
  static void ReorderPosteriorExecList(NotNull<std::vector<CNodePtr> *> node_list);
  // get fix output precision of cnode.
  static TypeId GetCNodeOutputPrecision(const AnfNodePtr &node);
  // get fix output precision from prev node, input_idx is the input index of current node related to prev node.
  static TypeId GetPrevNodeOutputPrecision(const AnfNodePtr &node, size_t input_idx);
  static bool IsDynamicShape(const AnfNodePtr &node);
  static bool HasDynamicShapeFlag(const PrimitivePtr &prim);
  static bool IsCondControlKernel(const CNodePtr &node);
  static bool IsIndependentNode(const CNodePtr &node);
  static bool GetBooleanAttr(const AnfNodePtr &node, const std::string &attr);
  static void GetRealDynamicShape(const std::vector<size_t> &shape, NotNull<std::vector<int64_t> *> dynamic_shape);
  static std::vector<int64_t> GetInputMaxShape(const AnfNodePtr &anf_node, size_t index);
  static std::vector<int64_t> GetInputMinShape(const AnfNodePtr &anf_node, size_t index);
  static std::vector<int64_t> GetOutputMaxShape(const AnfNodePtr &anf_node, size_t index);
  static std::vector<int64_t> GetOutputMinShape(const AnfNodePtr &anf_node, size_t index);
  static bool IsNodeDynamicShape(const AnfNodePtr &node);
  static void InferShape(const CNodePtr &node, std::map<uint32_t, tensor::TensorPtr> *depend_tensors = nullptr);
  static std::vector<size_t> GetInputRealDeviceShapeIfExist(const AnfNodePtr &anf_node, size_t index);
  static std::vector<size_t> GetOutputRealDeviceShapeIfExist(const AnfNodePtr &anf_node, size_t index);
  // Find real input nodes.
  static void GetAllFatherRealNode(const AnfNodePtr &anf_node, std::vector<AnfNodePtr> *result,
                                   std::set<AnfNodePtr> *visited);
  static void GetAllVisitedCNode(const CNodePtr &cnode, std::vector<AnfNodePtr> *used_kernels,
                                 std::set<AnfNodePtr> *visited);
  static void InsertMakeTupleForOutput(const NotNull<KernelGraphPtr> &root_graph);
  static AnfNodeIndexSet GetUpdateStateUsers(const FuncGraphManagerPtr &manager, const AnfNodePtr &node);
  // Get node real inputs, skip `MakeTuple`, `TupleGetItem`, `Depend`, `Load`, `UpdateState` etc.
  static void GetRealInputs(const AnfNodePtr &anf_node, std::vector<session::KernelWithIndex> *inputs);
  // Check whether tensors need broadcast or not.
  static bool IsTensorBroadcast(const std::vector<size_t> &lhs, const std::vector<size_t> &rhs);
  // Calc tensor size in byte.
  template <typename T>
  static size_t TensorSizeInByte(const std::vector<size_t> &shape) {
    size_t result = sizeof(T);
    for (size_t i = 0; i < shape.size(); i++) {
      result *= shape[i];
    }
    return result;
  }
  static bool IsOneOfPrimitiveCNode(const AnfNodePtr &node, const PrimitiveSet &prim_set);

  // Judge a control operator need be compiled into kernel graph rather than be cut into single op and
  // executed in vm. For example, the operator "bprop_cut" will be compiled into kernel graph and be launch
  // in backend in PyNative mode.
  static bool IsControlOpExecInBackend(const AnfNodePtr &node);

  static bool IsNodeInputContainMonad(const AnfNodePtr &node);
  // Save inputs/outputs/workspace address in kernel_mod.
  static void CacheAddrForGraph(const KernelGraphPtr &kernel_graph);
  static void CacheAddrForKernel(const AnfNodePtr &node, kernel::KernelMod *kernel_mod);
  static void CacheAddrForAtomicClean(const AnfNodePtr &node, kernel::KernelMod *kernel_mod);
};
}  // namespace session
using AnfAlgo = session::AnfRuntimeAlgorithm;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_ANF_RUNTIME_ALGORITHM_H
