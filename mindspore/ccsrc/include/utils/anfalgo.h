/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_ANFALGO_H
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_ANFALGO_H

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "base/base.h"
#include "include/utils/contract.h"
#include "include/utils/visible.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "primitive/array_op_name.h"
#include "primitive/other_op_name.h"
#include "primitive/sequence_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace common {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

class COMMON_EXPORT AnfAlgo {
 public:
  // get real input node of tuple_get_item
  static AnfNodePtr GetTupleGetItemRealInput(const CNodePtr &tuple_get_item);
  static size_t GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item);
  // get input_anf_node's real kernel by recurse
  static KernelWithIndex VisitKernel(const AnfNodePtr &anf_node, size_t index);
  static KernelWithIndex VisitKernelWithReturnType(
    const AnfNodePtr &anf_node, size_t index, bool skip_nop_node = false,
    const std::vector<PrimitivePtr> &return_types = {prim::kPrimMakeTuple},
    abstract::AbstractBasePtr *abstract = nullptr, bool is_index_valid = false);

  // Skip the monad node to get the real node.
  static KernelWithIndex FetchRealNodeSkipMonadControl(const KernelWithIndex &node_with_index);

  static std::vector<AnfNodePtr> GetAllOutput(const AnfNodePtr &node,
                                              const std::vector<PrimitivePtr> &return_types = {});
  static std::vector<KernelWithIndex> GetAllOutputIndexByReturnTypes(const AnfNodePtr &node,
                                                                     const std::vector<PrimitivePtr> &return_types = {},
                                                                     bool need_make_tuple = false);
  static std::vector<KernelWithIndex> GetAllOutputWithIndex(const AnfNodePtr &node,
                                                            const std::vector<PrimitivePtr> &return_types = {});
  static std::vector<KernelWithIndex> GetAllOutputWithOutMonadAndParameter(const AnfNodePtr &node);
  // get cnode primitive
  static AnfNodePtr GetCNodePrimitiveNode(const CNodePtr &node);
  static void SetNodeInput(const CNodePtr &node, const AnfNodePtr &input_node, size_t index);
  static PrimitivePtr GetCNodePrimitive(const AnfNodePtr &node);
  // Get cnode primitive attr.
  static ValuePtr GetCNodePrimitiveAttr(const AnfNodePtr &node, const std::string &key) {
    const auto &primitive = GetCNodePrimitive(node);
    return primitive != nullptr ? primitive->GetAttr(key) : nullptr;
  }
  // check whether anf node is a node of 'primitive_type',such as make_tuple is a cnode of kPrimMakeTuple
  static bool CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type);
  // get cnode primitive
  static FuncGraphPtr GetCNodeFuncGraphPtr(const AnfNodePtr &node);
  // get kernel_name of anf node
  static std::string GetCNodeName(const AnfNodePtr &node);
  static bool IsGetNextNode(const AnfNodePtr &node);
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
  // set attr of anf node safely(use a copy of primitive)
  static void SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node);
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
  static void EraseNodeAttr(const std::string &key, const AnfNodePtr &node);
  // get the num of inputs include monads for a cnode
  static size_t GetInputNum(const CNodePtr &cnode);
  // get the num of inputs exclude monads for real_kernel (which can be build and run in device)
  static size_t GetInputTensorNum(const AnfNodePtr &node);
  // get prev node output width output index
  static KernelWithIndex GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx, bool skip_nop_node = false);
  // get all the untuple real prev_nodes output
  static std::vector<KernelWithIndex> GetRealPrevNodesOutput(const AnfNodePtr &anf_node, size_t input_idx,
                                                             bool skip_nop_node = false);

  // get output shapes inferred by ME from input nodes.
  static ShapeVector GetOutputInferShape(const AnfNodePtr &node, size_t output_idx,
                                         bool is_real_squence_output = false);
  // get input shapes inferred by ME from input nodes.
  static ShapeVector GetPrevNodeOutputInferShape(const AnfNodePtr &node, size_t input_idx);
  // get output data type inferred by ME of anf node
  static TypePtr GetOutputInferType(const AnfNodePtr &node, size_t output_idx, bool is_real_tuple = false);
  static TypeId GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx);
  static TypeId GetOutputInferDataType(const TypePtr &type, size_t output_idx);
  // get output original data type from prev node,input_index is the input index of current node related to prev node
  static TypeId GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx);
  static TypePtr GetPrevNodeOutputInferType(const AnfNodePtr &node, size_t input_idx);
  // for tuple condition
  static std::vector<TypeId> GetRealPrevNodesOutputInferDataType(const AnfNodePtr &node, size_t input_idx);
  // set infer shapes and types of anf node
  static void SetOutputInferTypeAndShape(const std::vector<TypeId> &types, const std::vector<ShapeVector> &shapes,
                                         AnfNode *node, bool disable_dynamic_len = false);
  // set output shape ptr
  static void SetOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                          const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node);

  static void SetSingleOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                                const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node);

  static void CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node);
  // checkout whether the anf node is a graph kernel.
  static bool IsGraphKernel(const AnfNodePtr &node);
  // checkout whether the anf node is an inner node of graph kernel.
  static bool IsNodeInGraphKernel(const AnfNodePtr &node);
  // check parameter is weight or data
  static bool IsParameterWeight(const ParameterPtr &node);
  // Check whether the cnode update parameter
  static bool IsUpdateParameterKernel(const CNodePtr &node);
  static AnfNodePtr GetInputNode(const CNodePtr &node, size_t index);
  // Return true if it is either compute communication fusion operator or pure communication operator
  static bool IsCommunicationOp(const std::string &prim_name);
  static bool IsCommunicationOp(const AnfNodePtr &node);
  // Return true if it is a compute communication fusion operator
  static bool IsCommunicationFusionOp(const std::string &kernel_name);
  static bool IsCommunicationFusionOp(const AnfNodePtr &node);
  // Return true if it is a pure communication operator
  static bool IsNaiveCommunicationOp(const std::string &kernel_name);
  static bool IsNaiveCommunicationOp(const AnfNodePtr &node);
  static bool IsLcclCommunicationOp(const AnfNodePtr &node);
  static bool IsFusedCommunicationOp(const AnfNodePtr &node);
  static bool IsInplaceNode(const mindspore::AnfNodePtr &kernel, const string &type);
  static bool IsGetNext(const NotNull<AnfNodePtr> &node);
  static bool IsNeedSkipNopOpAddr(const AnfNodePtr &node);
  static FuncGraphPtr GetValueNodeFuncGraph(const AnfNodePtr &node);
  static bool IsScalarInput(const CNodePtr &cnode, size_t index);
  static bool IsScalarOutput(const CNodePtr &cnode, size_t index);
  static void ReorderExecList(NotNull<std::vector<CNodePtr> *> node_list);
  static void ReorderPosteriorExecList(NotNull<std::vector<CNodePtr> *> node_list);

  static std::string GetMoveToDstStr(const AnfNodePtr &node);
  static bool IsNodeInputDynamicShape(const CNodePtr &anf_node_ptr);
  static bool IsNodeOutputDynamicShape(const AnfNodePtr &node);
  static bool IsDynamicShape(const AnfNodePtr &node);
  static bool IsDynamicRankNode(const AnfNodePtr &node);
  static bool IsDynamicValue(const AnfNodePtr &node);
  static bool IsDynamic(const AnfNodePtr &node);
  static bool IsDynamicShapeFuncGraph(const FuncGraphPtr &func_graph);
  static bool IsNodeInputDynamicRank(const CNodePtr &anf_node_ptr);
  static bool IsNodeOutputDynamicRank(const AnfNodePtr &node);
  static bool IsOutputAnchorDynamicRank(const AnfNodePtr &node, size_t idx);
  static bool IsCondControlKernel(const CNodePtr &node);
  static bool GetBooleanAttr(const AnfNodePtr &node, const std::string &attr);
  static std::optional<string> GetDumpFlag(const AnfNodePtr &node);
  static std::vector<int64_t> GetOutputMaxShape(const AnfNodePtr &anf_node, size_t index);
  static bool IsHostKernel(const CNodePtr &kernel_node);
  // Used to check whether an AnfNode is a Summary Node.
  static bool IsSummaryNode(const AnfNodePtr &node);
  static bool IsAKGSparseOP(const AnfNodePtr &cnode);
  static std::string GetGraphSplitGroup(const AnfNodePtr &node);
  static AnfNodeIndexSet GetUpdateStateUsers(const FuncGraphManagerPtr &manager, const AnfNodePtr &node);
  // Get node real inputs, skip `MakeTuple`, `TupleGetItem`, `Depend`, `Load`, `UpdateState` etc.
  static void GetRealInputs(const AnfNodePtr &node, std::vector<KernelWithIndex> *inputs);
  // Check whether tensors need broadcast or not.
  template <typename T>
  static inline bool IsTensorBroadcast(const std::vector<T> &lhs, const std::vector<T> &rhs) {
    if (lhs.size() != rhs.size()) {
      return true;
    }
    for (size_t i = 0; i < lhs.size(); i++) {
      if (lhs[i] != rhs[i]) {
        return true;
      }
    }
    return false;
  }

  // Calc tensor size in byte.
  template <typename T>
  static size_t TensorSizeInByte(const std::vector<int64_t> &shape) {
    return sizeof(T) * SizeOf(shape);
  }

  template <typename T>
  static size_t TensorSizeInByte(const std::vector<size_t> &shape) {
    size_t res = sizeof(T);
    res = std::accumulate(shape.begin(), shape.end(), res, std::multiplies<size_t>());

    return res;
  }

  // Judge a control operator need be compiled into kernel graph rather than be cut into single op and
  // executed in vm. For example, the operator "bprop_cut" will be compiled into kernel graph and be launch
  // in backend in PyNative mode.
  static bool IsBpropCutOpExecInBackend(const AnfNodePtr &node);
  // Check whether a cnode has a monad input.
  static bool HasMonadInput(const AnfNodePtr &node);
  // Check if node has none input after IR fusion.
  static bool IsNoneInput(const AnfNodePtr &node, size_t index);
  // Check whether node is a call node, call nodes are those cnodes whose first input is not primitive node.
  static bool IsCallNode(const AnfNodePtr &node);
  // Get the output number according to abstract, when there is a tuple in abstract, it needs to get recursively.
  static size_t GetOutputNumByAbstract(const AbstractBasePtr &node_abstract);
  // Get attr groups
  static int64_t GetAttrGroups(const AnfNodePtr &node, size_t index);

  static inline bool IsAllgather(const CNodePtr &cnode) { return GetCNodeName(cnode) == kAllGatherOpName; }

  static bool IsFusion(const CNodePtr &cnode);

  static inline bool IsFromParallelOptimizer(const CNodePtr &cnode) {
    auto primitive = GetCNodePrimitive(cnode);
    return (primitive != nullptr) && primitive->instance_name().find("parallel_optimizer") != std::string::npos;
  }

  static bool IsRecompute(const CNodePtr &cnode);

  // Check whether the node has Ref abstract.
  static inline bool HasAbstractRef(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    auto &abs = node->abstract();
    return (abs != nullptr) && abs->isa<abstract::AbstractRefTensor>();
  }

  // Check whether the sequence node has Ref abstract.
  static inline bool SequenceHasAbstractRef(const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    auto &abs = node->abstract();
    if ((abs != nullptr) && (abs->isa<abstract::AbstractSequence>())) {
      auto abs_seq = abs->cast_ptr<abstract::AbstractSequence>();
      const auto &elements = abs_seq->elements();
      return std::any_of(elements.begin(), elements.end(), [](const AbstractBasePtr &element) {
        return (element != nullptr) && element->isa<abstract::AbstractRefTensor>();
      });
    }
    return false;
  }

  static tensor::TensorPtr SequenceToTensor(const ValuePtr &value);

  // Get the real output node and indexes of get item, make tuple, depend, load.
  static AnfNodePtr GetTupleIndexes(const AnfNodePtr &node, std::vector<size_t> *const index_stack);
  static bool IsNopNode(const AnfNodePtr &node);
  static bool IsViewNode(const AnfNodePtr &node);
  static bool CheckStridedSliceForwardOrBackWardIsNopNode(const CNodePtr &cnode);
  template <typename T>
  static bool CheckAbsType(const AnfNodePtr &node);
  static bool CheckAbsSparseTensor(const AnfNodePtr &node);
  static bool CheckAbsSparseTensor(const abstract::AbstractBasePtr &abs);
  static TypeId GetSparseTypeIdAt(const AnfNodePtr &node, size_t idx);

  static std::string GetTensorValueString(const tensor::TensorPtr &tensor);

  static bool IsNodeMutableScalar(const AnfNodePtr &node);
  static bool IsDynamicSequence(const AnfNodePtr &node);
  static bool IsAnyTypeOutput(const AnfNodePtr &node);
  static bool IsAnyTypeInput(const std::vector<AnfNodePtr> &inputs);
  static bool HasTupleInput(const CNodePtr &node);
  static bool HasDynamicTupleInput(const CNodePtr &node);
  static bool IsReduceOp(const std::string &op_name);
  static bool IsTypeTransformOp(const std::string &op_name);
  // Get the element shape of dynamic sequence shape.
  static abstract::BaseShapePtr GetDynamicSequenceShape(const AnfNodePtr &node, size_t output_idx);
  // Fetch the sub abstract from the top abstract by the index.
  static abstract::AbstractBasePtr FetchAbstractByIndex(const AbstractBasePtr &abstract, size_t index);

  static std::string GetInputName(const CNodePtr &origin_op, size_t input_index);
  static bool IsNoOuputNode(const AnfNodePtr &node);
  static ValuePtr ValueToScalar(const ValuePtr &value, TypeId type_id);
  static std::vector<ValuePtr> TransformVectorRefToMultiValue(const VectorRef &base_ref);
  static bool HasIncorporateCallNode(const CNodePtr &cnode);
  static bool IsDynamicGraph(const FuncGraphPtr &func_graph);
  static CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &tuple_inputs);
  static void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node,
                           const FuncGraphManagerPtr &manager, const FuncGraphPtr &root,
                           const std::string &attr_tag = "", const size_t post_node_input_index = 1);
  static bool IsNeededOverlapComm(const CNodePtr &cnode, const std::string &pp_1f1b_value);
  static AnfNodePtr GetInputNode(const AnfNodePtr &node,
                                 std::function<std::pair<bool, size_t>(const CNodePtr &)> check_filter);
  static bool IsNeededShape(const CNodePtr &cnode);
  static bool IsMonadType(const TypeId &type_id);
  // if graph output is valuenode or parameter, used to skip run and construct output
  static bool IsGraphOutputValueNodeOrParameter(const AnfNodePtr &graph_output, const VectorRef &args,
                                                VectorRef *outputs);
  // charge if the node's output is a feature map output
  static bool IsFeatureMapOutput(const AnfNodePtr &node);
  // charge if the node's input is from a feature map output
  static bool IsFeatureMapInput(const AnfNodePtr &node, size_t input_index);
};
}  // namespace common
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_ANFALGO_H
