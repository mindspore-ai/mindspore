/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <tuple>
#include <utility>
#include <memory>
#include <map>
#include <functional>
#include <optional>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/dtype.h"
#include "base/base.h"
#include "ir/primitive.h"
#include "ir/kernel_info_dev.h"
#include "mindspore/core/ops/core_ops.h"
#include "include/common/utils/contract.h"
#include "utils/anf_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/visible.h"

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
    abstract::AbstractBasePtr *abstract = nullptr);

  // Skip the monad node to get the real node.
  static KernelWithIndex FetchRealNodeSkipMonadControl(const KernelWithIndex &node_with_index);

  static std::vector<AnfNodePtr> GetAllOutput(const AnfNodePtr &node,
                                              const std::vector<PrimitivePtr> &return_types = {});
  static std::vector<KernelWithIndex> GetAllOutputIndexByReturnTypes(const AnfNodePtr &node,
                                                                     const std::vector<PrimitivePtr> &return_types = {},
                                                                     bool need_make_tuple = false);
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
  // get prev node output width output index has tuplegetitem
  static bool IsPrevNodeHasTupleGetItem(const AnfNodePtr &anf_node, size_t input_idx, bool skip_nop_node = false);
  // get prev node output width output index
  static KernelWithIndex GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx, bool skip_nop_node = false);
  // get all the untuple real prev_nodes output
  static std::vector<KernelWithIndex> GetRealPrevNodesOutput(const AnfNodePtr &anf_node, size_t input_idx,
                                                             bool skip_nop_node = false);

  // get output shapes inferred by ME from input nodes.
  static ShapeVector GetOutputInferShape(const AnfNodePtr &node, size_t output_idx,
                                         bool is_real_squence_output = false);
  static ShapeVector GetOutputInferShape(const AnfNodePtr &node, const abstract::BaseShapePtr &base_shape,
                                         size_t output_idx, bool is_real_squence_output = false);
  // get input shapes inferred by ME from input nodes.
  static ShapeVector GetPrevNodeOutputInferShape(const AnfNodePtr &node, size_t input_idx);
  // get output data type inferred by ME of anf node
  static TypeId GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx);
  static TypeId GetOutputInferDataType(const TypePtr &type, size_t output_idx);
  // get output original data type from prev node,input_index is the input index of current node related to prev node
  static TypeId GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx);
  // for tuple condition
  static std::vector<TypeId> GetRealPrevNodesOutputInferDataType(const AnfNodePtr &node, size_t input_idx);
  // set infer shapes and types of anf node
  static void SetOutputInferTypeAndShape(const std::vector<TypeId> &types, const std::vector<ShapeVector> &shapes,
                                         AnfNode *node, bool disable_dynamic_len = false);
  static void SetScalarTupleOutputInferType(const std::vector<TypeId> &types, const std::vector<ShapeVector> &shapes,
                                            const AnfNodePtr &node);
  // set output shape ptr
  static void SetOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                          const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node);
  static void CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node);
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
  // Check whether the cnode update parameter
  static bool IsUpdateParameterKernel(const CNodePtr &node);
  static AnfNodePtr GetInputNode(const CNodePtr &node, size_t index);
  static bool IsCommunicationOp(const AnfNodePtr &node);
  static bool IsDtypeFormatSensitiveOp(const AnfNodePtr &node);
  static bool IsFusedCommunicationOp(const AnfNodePtr &node);
  static bool IsInplaceNode(const mindspore::AnfNodePtr &kernel, const string &type);
  static bool IsGetNext(const NotNull<AnfNodePtr> &node);
  static bool IsNeedSkipNopOpAddr(const AnfNodePtr &node);
  static bool IsNeedSkipNopOpExecution(const AnfNodePtr &node);
  static FuncGraphPtr GetValueNodeFuncGraph(const AnfNodePtr &node);
  static bool IsSwitchCall(const CNodePtr &call_node);
  static bool IsScalarInput(const CNodePtr &cnode, size_t index);
  static bool IsScalarOutput(const CNodePtr &cnode, size_t index);
  static void ReorderExecList(NotNull<std::vector<CNodePtr> *> node_list);
  static void ReorderPosteriorExecList(NotNull<std::vector<CNodePtr> *> node_list);
  // get fix output precision of cnode.
  static TypeId GetCNodeOutputPrecision(const AnfNodePtr &node);
  // get fix output precision from prev node, input_idx is the input index of current node related to prev node.
  static TypeId GetPrevNodeOutputPrecision(const AnfNodePtr &node, size_t input_idx);
  static bool IsNodeInputDynamicShape(const CNodePtr &anf_node_ptr);
  static bool IsNodeOutputDynamicShape(const AnfNodePtr &node);
  static bool IsDynamicShape(const AnfNodePtr &node);
  static bool IsDynamicRankNode(const AnfNodePtr &node);
  static bool IsNodeInputDynamicRank(const CNodePtr &anf_node_ptr);
  static bool IsNodeOutputDynamicRank(const AnfNodePtr &node);
  static bool IsInputAnchorDynamicRank(const AnfNodePtr &node, size_t idx);
  static bool IsOutputAnchorDynamicRank(const AnfNodePtr &node, size_t idx);
  static bool HasDynamicShapeFlag(const PrimitivePtr &prim);
  static bool IsCondControlKernel(const CNodePtr &node);
  static bool GetBooleanAttr(const AnfNodePtr &node, const std::string &attr);
  static std::optional<string> GetDumpFlag(const AnfNodePtr &node);
  static void GetRealDynamicShape(const std::vector<size_t> &shape, NotNull<std::vector<int64_t> *> dynamic_shape);
  static std::vector<int64_t> GetOutputMaxShape(const AnfNodePtr &anf_node, size_t index);
  static bool IsHostKernel(const CNodePtr &kernel_node);
  static void AddArgList(AbstractBasePtrList *args_spec_list, const AnfNodePtr &real_input, size_t real_input_index);
  // Find real input nodes.
  static void GetAllFatherRealNode(const AnfNodePtr &anf_node, std::vector<AnfNodePtr> *result,
                                   std::set<AnfNodePtr> *visited);
  static void GetAllVisitedCNode(const CNodePtr &node, std::vector<AnfNodePtr> *used_kernels,
                                 std::set<AnfNodePtr> *visited);
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
  static bool IsControlOpExecInBackend(const AnfNodePtr &node);

  static bool IsNodeInputContainMonad(const AnfNodePtr &node);
  // Check if node is non-task op.
  static bool IsNonTaskOp(const CNodePtr &node);
  // Check if node has none input after IR fusion.
  static bool IsNoneInput(const AnfNodePtr &node, size_t index);
  // Check whether node is a call node, call nodes are those cnodes whose first input is not primitive node.
  static bool IsCallNode(const AnfNodePtr &node);
  // Get the output number according to abstract, when there is a tuple in abstract, it needs to get recursively.
  static size_t GetOutputNumByAbstract(const AbstractBasePtr &node_abstract);
  // Get attr groups
  static int64_t GetAttrGroups(const AnfNodePtr &node, size_t index);

  static inline bool IsAllgather(const CNodePtr &cnode) { return GetCNodeName(cnode) == kAllGatherOpName; }

  static inline bool IsFusion(const CNodePtr &cnode) {
    return HasNodeAttr(kAttrFusion, cnode) && GetNodeAttr<int64_t>(cnode, kAttrFusion) > 0;
  }

  static inline bool IsFromParallelOptimizer(const CNodePtr &cnode) {
    auto primitive = GetCNodePrimitive(cnode);
    return (primitive != nullptr) && primitive->instance_name().find("parallel_optimizer") != std::string::npos;
  }

  static inline bool IsRecompute(const CNodePtr &cnode) {
    auto attr_dup = cnode->GetAttr(kAttrDuplicated);
    return attr_dup != nullptr && GetValue<bool>(attr_dup);
  }

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

  // Get the real output node and indexes of get item, make tuple, depend, load.
  static AnfNodePtr GetTupleIndexes(const AnfNodePtr &node, std::vector<size_t> *const index_stack);
  static bool IsNopNode(const AnfNodePtr &node);

  template <typename T>
  static bool CheckAbsType(const AnfNodePtr &node);
  static bool CheckAbsSparseTensor(const AnfNodePtr &node);
  static bool CheckAbsSparseTensor(const abstract::AbstractBasePtr &abs);
  static TypeId GetSparseTypeIdAt(const AnfNodePtr &node, size_t idx);

  static std::string GetTensorValueString(const tensor::TensorPtr &tensor);
  static abstract::AbstractBasePtr GetNodeAbstractByIndex(const AnfNodePtr &node, size_t index);

  // Get jit level from func_graph
  static std::string GetJitLevel(const FuncGraphPtr &func_graph);

  static bool IsDynamicSequence(const AnfNodePtr &node);
  static bool HasTupleInput(const CNodePtr &node);
  static bool HasDynamicTupleInput(const CNodePtr &node);
  static bool IsReduceOp(const std::string &op_name);

  // Get the element shape of dynamic sequence shape.
  static abstract::BaseShapePtr GetDynamicSequenceShape(const AnfNodePtr &node, size_t output_idx);
};
}  // namespace common
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_ANFALGO_H
