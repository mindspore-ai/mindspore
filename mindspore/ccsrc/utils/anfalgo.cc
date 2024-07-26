/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "include/common/utils/anfalgo.h"
#include <memory>
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <complex>
#include "mindapi/base/shape_vector.h"
#include "ops/ascend_op_name.h"
#include "ops/nn_optimizer_op_name.h"
#include "ops/lite_op_name.h"
#include "ops/structure_ops.h"
#include "ops/sequence_ops.h"
#include "ops/other_ops.h"
#include "ops/nn_ops.h"
#include "ops/math_ops.h"
#include "ops/array_ops.h"
#include "ops/arithmetic_ops.h"
#include "ops/framework_ops.h"
#include "ops/op_utils.h"
#include "ops/op_def.h"
#include "ops/auto_generate/gen_ops_primitive.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "include/common/utils/utils.h"
#include "utils/shape_utils.h"
#include "utils/trace_base.h"
#include "utils/anf_utils.h"
#include "include/common/utils/parallel_context.h"
#include "utils/ms_context.h"
#include "pybind_api/ir/primitive_py.h"
#include "kernel/kernel_build_info.h"
#include "include/backend/anf_runtime_algorithm.h"

namespace mindspore {
namespace common {
using abstract::AbstractSparseTensor;
using abstract::AbstractTensor;
using abstract::AbstractTuple;

namespace {
constexpr size_t kNopNodeRealInputIndex = 1;
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

const PrimitiveSet expand_prims = {prim::kPrimMakeTuple};
const std::set<std::string> kNodeTupleOutSet = {kMakeTupleOpName, kGetNextOpName};

void GetRealOutputRecursively(const AnfNodePtr &node, size_t output_index, std::vector<KernelWithIndex> *inputs) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<ValueNode>() || node->isa<Parameter>()) {
    return inputs->push_back(std::make_pair(node, 0));
  }

  // Skip control node
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend) || AnfAlgo::CheckPrimitiveType(node, prim::kPrimLoad) ||
      AnfAlgo::CheckPrimitiveType(node, prim::kPrimUpdateState)) {
    return GetRealOutputRecursively(node->cast<CNodePtr>()->input(kRealInputIndexInDepend), 0, inputs);
  }

  // Bypass TupleGetItem
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimTupleGetItem)) {
    auto tuple_get_item = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_item);
    auto input = AnfAlgo::GetTupleGetItemRealInput(tuple_get_item);
    auto index = AnfAlgo::GetTupleGetItemOutIndex(tuple_get_item);
    // Conceal MakeTuple + TupleGetItem pair.
    if (AnfAlgo::CheckPrimitiveType(input, prim::kPrimMakeTuple)) {
      auto make_tuple = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(make_tuple);
      auto real_input = AnfAlgo::GetInputNode(make_tuple, index);
      return GetRealOutputRecursively(real_input, 0, inputs);
    }

    // Skip TupleGetItem.
    return GetRealOutputRecursively(input, index, inputs);
  }

  // Flatten MakeTuple inputs.
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimMakeTuple)) {
    auto make_tuple = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    size_t input_num = AnfAlgo::GetInputTensorNum(make_tuple);
    for (size_t input_index = 0; input_index < input_num; ++input_index) {
      auto input_node = AnfAlgo::GetInputNode(make_tuple, input_index);
      GetRealOutputRecursively(input_node, 0, inputs);
    }
    return;
  }

  return inputs->push_back(std::make_pair(node, output_index));
}

bool IsMultiLayerTuple(const abstract::AbstractBasePtr &abstract) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abstract);
  if (sequence_abstract->dynamic_len()) {
    return false;
  }
  return std::any_of(sequence_abstract->elements().begin(), sequence_abstract->elements().end(),
                     [](const abstract::AbstractBasePtr &sub_abstract) {
                       return sub_abstract != nullptr && sub_abstract->isa<abstract::AbstractSequence>();
                     });
}

std::vector<KernelWithIndex> GetAllOutputWithIndexInner(const AnfNodePtr &node,
                                                        const std::vector<PrimitivePtr> &return_types) {
  MS_EXCEPTION_IF_NULL(node);
  MS_LOG(DEBUG) << "Output node: " << node->fullname_with_scope();
  std::vector<KernelWithIndex> ret;
  std::vector<KernelWithIndex> ret_empty;
  // The MakeTuple/MakeSparse node need expand and recurse.
  if (IsOneOfPrimitiveCNode(node, expand_prims)) {
    auto make_tuple = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->size(); i++) {
      auto make_tuple_output = GetAllOutputWithIndexInner(make_tuple->input(i), return_types);
      (void)std::copy(make_tuple_output.begin(), make_tuple_output.end(), std::back_inserter(ret));
    }
    return ret;
  }
  // The depend node need get the real node.
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    auto depend_node = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(depend_node);
    auto real_output = GetAllOutputWithIndexInner(depend_node->input(kRealInputIndexInDepend), return_types);
    (void)std::copy(real_output.begin(), real_output.end(), std::back_inserter(ret));
    return ret;
  }

  // Value node need get all the elements.
  if (node->isa<ValueNode>()) {
    auto value = node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<ValueSequence>()) {
      auto value_tuple = value->cast<ValueSequencePtr>();
      auto value_tuple_size = CountValueNum(value_tuple);
      for (size_t i = 0; i < value_tuple_size; ++i) {
        (void)ret.emplace_back(node, i);
      }
    } else {
      (void)ret.emplace_back(node, 0);
    }
    MS_LOG(DEBUG) << "Output value node: " << node->fullname_with_scope() << ", value num: " << ret.size();
    return ret;
  }

  // Output num must be exactly equal to the number of outputs of the node.
  size_t outputs_num = 1;
  if (AnfUtils::IsRealCNodeKernel(node)) {
    if (node->abstract() != nullptr &&
        (common::AnfAlgo::IsDynamicSequence(node) || IsMultiLayerTuple(node->abstract()))) {
      outputs_num = common::AnfAlgo::GetOutputNumByAbstract(node->abstract());
    } else {
      outputs_num = AnfUtils::GetOutputTensorNum(node);
    }
    MS_LOG(DEBUG) << "Output num:" << outputs_num << " for node:" << node->DebugString();
  }
  // Call node maybe a real cnode and the unreal node cannot get output num exactly, so we should get
  // output num from abstract again. For example the TupleGetItem/Makeple multi-level nesting:
  // '''G = op()  ---> Assume that the output of G is a multi-member tuple
  //    A = MakeTuple(E, F, G)
  //    B = MakeTuple(H, A)
  //    C = TupleGetItem(B, 1) ---> Euqal the A
  //    D = TupleGetItem(C, 2)  ---> VisitKernel will return the {G, 0}, but expect the whole G with all the members
  //    return D'''
  if (common::AnfAlgo::IsCallNode(node) || (!AnfUtils::IsRealCNodeKernel(node))) {
    MS_EXCEPTION_IF_NULL(node->abstract());
    outputs_num = AnfAlgo::GetOutputNumByAbstract(node->abstract());
  }

  // The output may be the tuple of node, so need visit all the outputs of node.
  // Since output num represents the number of all outputs of node, only one output is obtained per loop.
  for (size_t i = 0; i < outputs_num; ++i) {
    // Maybe this scene: tupleGetItem + depend + makeTuple, can be done correctly in VisitKernelWithReturnType.
    // The output may be updataState/load node for connecting dependencies between subgraphs.
    auto output_with_index = AnfAlgo::VisitKernelWithReturnType(
      node, i, false, {prim::kPrimMakeTuple, prim::kPrimUpdateState, prim::kPrimLoad}, nullptr, true);
    MS_EXCEPTION_IF_NULL(output_with_index.first);

    // The MakeTuple/MakeSparse node need recurse.
    if (IsOneOfPrimitiveCNode(output_with_index.first, expand_prims)) {
      auto output_vector = GetAllOutputWithIndexInner(output_with_index.first, return_types);
      if (output_vector.size() <= output_with_index.second) {
        MS_LOG(INTERNAL_EXCEPTION) << "Invalid index:" << output_with_index.second
                                   << " for outputs of node:" << output_with_index.first->DebugString();
      }
      (void)ret.emplace_back(output_vector[output_with_index.second]);
      continue;
    }

    // The InitDataSetQueue node has no output.
    if (AnfAlgo::CheckPrimitiveType(output_with_index.first, prim::kPrimInitDataSetQueue)) {
      return ret_empty;
    }

    MS_LOG(DEBUG) << "Output node: " << output_with_index.first->fullname_with_scope()
                  << " with output index: " << output_with_index.second;
    ret.push_back(output_with_index);
  }
  return ret;
}

bool IsNodeDynamicShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Node is not a cnode";
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  auto in_dynamic = AnfAlgo::IsNodeInputDynamicShape(cnode);
  auto out_dynamic = AnfAlgo::IsNodeOutputDynamicShape(cnode);
  if (in_dynamic && !AnfAlgo::HasNodeAttr(kAttrInputIsDynamicShape, cnode)) {
    AnfAlgo::SetNodeAttrSafely(kAttrInputIsDynamicShape, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Set Input Dynamic Shape Attr to Node:" << cnode->fullname_with_scope()
                  << " debug string:" << cnode->DebugString();
  }
  if (out_dynamic && !AnfAlgo::HasNodeAttr(kAttrOutputIsDynamicShape, cnode)) {
    AnfAlgo::SetNodeAttrSafely(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Set Output Dynamic Shape Attr to Node:" << cnode->fullname_with_scope()
                  << " debug string:" << cnode->DebugString();
  }
  if (IsPrimitiveCNode(node, prim::kPrimPyExecute) && node->abstract()->isa<abstract::AbstractSequence>()) {
    AnfAlgo::SetNodeAttrSafely(kAttrOutputIsDynamicShape, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Set Output Dynamic Shape Attr to Node:" << cnode->fullname_with_scope();
    return true;
  }
  return in_dynamic || out_dynamic;
}
}  // namespace

AnfNodePtr AnfAlgo::GetTupleGetItemRealInput(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(INTERNAL_EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
  }
  return tuple_get_item->input(kRealInputNodeIndexInTupleGetItem);
}

size_t AnfAlgo::GetTupleGetItemOutIndex(const CNodePtr &tuple_get_item) {
  MS_EXCEPTION_IF_NULL(tuple_get_item);
  if (tuple_get_item->size() != kTupleGetItemInputSize) {
    MS_LOG(INTERNAL_EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
  }
  auto output_index_value_node = tuple_get_item->input(kInputNodeOutputIndexInTupleGetItem);
  MS_EXCEPTION_IF_NULL(output_index_value_node);
  auto value_node = output_index_value_node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto idx = value->isa<Int64Imm>() ? GetValue<int64_t>(value) : GetValue<int>(value);
  return LongToSize(idx);
}

KernelWithIndex AnfAlgo::VisitKernel(const AnfNodePtr &anf_node, size_t index) {
  // this function was moved to AnfUtils.
  return AnfUtils::VisitKernel(anf_node, index);
}

namespace {
KernelWithIndex VisitKernelWithReturnTypeForTupleGetItem(const AnfNodePtr &anf_node, size_t index, bool skip_nop_node,
                                                         const std::vector<PrimitivePtr> &return_types,
                                                         abstract::AbstractBasePtr *abstract, bool is_index_valid) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!common::AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    MS_LOG(EXCEPTION) << "Invalid tuple get item node:" << anf_node->DebugString();
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->HasAttr(kAttrReplaceRealKernelInBackend)) {
    MS_LOG(INFO) << "cnode:" << cnode->DebugString() << " has replace flag";
    return KernelWithIndex(anf_node, index);
  }
  abstract::AbstractBasePtr abs = nullptr;
  auto item_with_index_tmp = common::AnfAlgo::VisitKernelWithReturnType(
    common::AnfAlgo::GetTupleGetItemRealInput(cnode), common::AnfAlgo::GetTupleGetItemOutIndex(cnode), skip_nop_node,
    return_types, &abs, true);
  if (IsOneOfPrimitiveCNode(item_with_index_tmp.first, expand_prims)) {
    MS_EXCEPTION_IF_NULL(item_with_index_tmp.first);
    auto make_tuple = item_with_index_tmp.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    const std::vector<AnfNodePtr> &make_tuple_inputs = make_tuple->inputs();
    size_t make_tuple_input_index = item_with_index_tmp.second + 1;
    if (make_tuple_input_index >= make_tuple_inputs.size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Index[" << make_tuple_input_index << "] out of range[" << make_tuple_inputs.size()
                                 << "].\nPlease check node: " << cnode->DebugString()
                                 << ".\nLine: " << trace::GetDebugInfoStr(cnode->debug_info())
                                 << ".\nAnd check node: " << make_tuple->DebugString()
                                 << ".\nLine: " << trace::GetDebugInfoStr(make_tuple->debug_info()) << ".";
    }
    return common::AnfAlgo::VisitKernelWithReturnType(make_tuple_inputs[make_tuple_input_index], index, skip_nop_node,
                                                      return_types);
  }
  if (common::AnfAlgo::IsCallNode(item_with_index_tmp.first) || item_with_index_tmp.first->isa<Parameter>()) {
    size_t real_index = item_with_index_tmp.second;
    if (abs == nullptr) {
      abs = item_with_index_tmp.first->abstract();
      real_index = 0;
    }
    MS_EXCEPTION_IF_NULL(abs);
    if (abs->isa<abstract::AbstractSequence>()) {
      auto tuple_abstract = abs->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(tuple_abstract);
      if (tuple_abstract->dynamic_len()) {
        return item_with_index_tmp;
      }
      auto sub_abstracts = tuple_abstract->elements();
      if (sub_abstracts.size() <= common::AnfAlgo::GetTupleGetItemOutIndex(cnode)) {
        MS_LOG(INTERNAL_EXCEPTION) << "Invalid index:" << common::AnfAlgo::GetTupleGetItemOutIndex(cnode)
                                   << " for abstract:" << abs->ToString();
      }
      for (size_t i = 0; i < common::AnfAlgo::GetTupleGetItemOutIndex(cnode); ++i) {
        MS_EXCEPTION_IF_NULL(sub_abstracts[i]);
        real_index += AnfAlgo::GetOutputNumByAbstract(sub_abstracts[i]);
      }
      if (abstract != nullptr) {
        (*abstract) = sub_abstracts[common::AnfAlgo::GetTupleGetItemOutIndex(cnode)];
        MS_EXCEPTION_IF_NULL((*abstract));
      } else {
        // In recursion of getitem node, the index of the first input of its real node is returned.
        // When the recursion ends, the outermost index needs to be accumulated.
        real_index += index;
      }
      return {item_with_index_tmp.first, real_index};
    }
  }
  if (is_index_valid) {
    if (anf_node->abstract() != nullptr && anf_node->abstract()->isa<abstract::AbstractSequence>()) {
      const auto &seq_abs = anf_node->abstract()->cast<abstract::AbstractSequencePtr>();
      MS_EXCEPTION_IF_NULL(seq_abs);
      if (!seq_abs->dynamic_len()) {
        return {anf_node, index};
      }
    }
  }
  return item_with_index_tmp;
}
}  // namespace

KernelWithIndex AnfAlgo::VisitKernelWithReturnType(const AnfNodePtr &anf_node, size_t index, bool skip_nop_node,
                                                   const std::vector<PrimitivePtr> &return_types,
                                                   abstract::AbstractBasePtr *abstract, bool is_index_valid) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (std::any_of(return_types.begin(), return_types.end(), [&anf_node](const PrimitivePtr &prim_type) -> bool {
        return CheckPrimitiveType(anf_node, prim_type);
      })) {
    return KernelWithIndex(anf_node, index);
  }
  if (!anf_node->isa<CNode>()) {
    return KernelWithIndex(anf_node, index);
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // TupleGetItem and SparseGetAttr needs to find real input
  if (CheckPrimitiveType(cnode, prim::kPrimTupleGetItem)) {
    return VisitKernelWithReturnTypeForTupleGetItem(anf_node, index, skip_nop_node, return_types, abstract,
                                                    is_index_valid);
  }
  if (AnfAlgo::CheckPrimitiveType(cnode, prim::kPrimUpdateState)) {
    return VisitKernelWithReturnType(cnode->input(kUpdateStateStateInput), index, skip_nop_node, return_types);
  }
  const PrimitiveSet follow_first_input_prims = {prim::kPrimDepend, prim::kPrimLoad, prim::kPrimDynamicLossScale};
  if (IsOneOfPrimitiveCNode(cnode, follow_first_input_prims)) {
    return VisitKernelWithReturnType(cnode->input(kRealInputIndexInDepend), index, skip_nop_node, return_types);
  }
  if (IsNopNode(cnode) && skip_nop_node) {
    return VisitKernelWithReturnType(cnode->input(kNopNodeRealInputIndex), 0, skip_nop_node, return_types);
  }
  return KernelWithIndex(anf_node, index);
}

KernelWithIndex AnfAlgo::FetchRealNodeSkipMonadControl(const KernelWithIndex &node_with_index) {
  MS_EXCEPTION_IF_NULL(node_with_index.first);
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> auto_monad_prims = {prim::kPrimDepend,
                                                                                              prim::kPrimLoad};
  if (IsOneOfPrimitiveCNode(node_with_index.first, auto_monad_prims)) {
    return common::AnfAlgo::VisitKernelWithReturnType(node_with_index.first, node_with_index.second, false);
  } else {
    return node_with_index;
  }
}

std::vector<AnfNodePtr> AnfAlgo::GetAllOutput(const AnfNodePtr &node, const std::vector<PrimitivePtr> &return_types) {
  std::vector<AnfNodePtr> ret;
  const auto &output_pair = GetAllOutputIndexByReturnTypes(node, return_types);
  std::transform(output_pair.begin(), output_pair.end(), std::back_inserter(ret),
                 [](const KernelWithIndex &ele) { return ele.first; });
  return ret;
}

std::vector<KernelWithIndex> AnfAlgo::GetAllOutputIndexByReturnTypes(const AnfNodePtr &node,
                                                                     const std::vector<PrimitivePtr> &return_types,
                                                                     bool need_make_tuple) {
  std::vector<KernelWithIndex> ret;
  auto return_prim_type = return_types;
  // if visited make_tuple should return back
  return_prim_type.push_back(prim::kPrimMakeTuple);
  auto item_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, false, return_prim_type);
  if (need_make_tuple) {
    ret.push_back(item_with_index);
  }
  if (AnfAlgo::CheckPrimitiveType(item_with_index.first, prim::kPrimMakeTuple)) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    auto make_tuple = item_with_index.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    for (size_t i = 1; i < make_tuple->size(); i++) {
      auto input_i_vector = GetAllOutputIndexByReturnTypes(make_tuple->input(i), return_types);
      (void)std::copy(input_i_vector.begin(), input_i_vector.end(), std::back_inserter(ret));
    }
    return ret;
  }
  ret.push_back(item_with_index);
  return ret;
}

size_t AnfAlgo::GetOutputNumByAbstract(const AbstractBasePtr &node_abstract) {
  MS_EXCEPTION_IF_NULL(node_abstract);
  size_t result = 0;

  if (!node_abstract->isa<abstract::AbstractSequence>() ||
      node_abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len() ||
      node_abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len_element_abs() != nullptr) {
    return 1;
  }

  auto tuple_abstract = node_abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  const auto &sub_abstracts = tuple_abstract->elements();
  for (const auto &sub_abstract : sub_abstracts) {
    MS_EXCEPTION_IF_NULL(sub_abstract);
    result += GetOutputNumByAbstract(sub_abstract);
  }
  return result;
}

std::vector<KernelWithIndex> AnfAlgo::GetAllOutputWithOutMonadAndParameter(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &graph_outputs = common::AnfAlgo::GetAllOutputWithIndex(node);
  std::vector<KernelWithIndex> real_output;
  for (const auto &node_with_index : graph_outputs) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    if (HasAbstractMonad(node_with_index.first) || node_with_index.first->isa<Parameter>() ||
        node_with_index.first->isa<ValueNode>()) {
      continue;
    }
    real_output.emplace_back(node_with_index);
  }
  return real_output;
}

std::vector<KernelWithIndex> AnfAlgo::GetAllOutputWithIndex(const AnfNodePtr &node,
                                                            const std::vector<PrimitivePtr> &return_types) {
  auto ret = GetAllOutputWithIndexInner(node, return_types);
  std::map<AnfNodePtr, size_t> value_node_index;

  // Unify the output of the front and back end to the ValueTuple
  for (auto &output_with_index : ret) {
    auto value_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(value_node);
    if (!value_node->isa<ValueNode>()) {
      continue;
    }
    if (value_node_index.find(value_node) == value_node_index.end() ||
        value_node_index[value_node] < output_with_index.second) {
      value_node_index[value_node] = output_with_index.second;
    } else {
      value_node_index[value_node]++;
      MS_LOG(DEBUG) << "Set output value node new index, value node: " << value_node->fullname_with_scope()
                    << ", original index: " << output_with_index.second
                    << ", new index:" << value_node_index[value_node];
      output_with_index.second = value_node_index[value_node];
    }
  }
  return ret;
}

bool AnfAlgo::CheckPrimitiveType(const AnfNodePtr &node, const PrimitivePtr &primitive_type) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return IsPrimitive(cnode->input(kAnfPrimitiveIndex), primitive_type);
}

FuncGraphPtr AnfAlgo::GetCNodeFuncGraphPtr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = cnode->input(kAnfPrimitiveIndex);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto value_node = attr_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  return value->cast<FuncGraphPtr>();
}

std::string AnfAlgo::GetCNodeName(const AnfNodePtr &node) {
  // this function was moved to AnfUtils.
  return AnfUtils::GetCNodeName(node);
}

bool AnfAlgo::IsGetNextNode(const AnfNodePtr &node) {
  auto node_name = AnfUtils::GetCNodeName(node);
  return node_name == kGetNextOpName || node_name == kDynamicGetNextV2OpName;
}

std::string AnfAlgo::GetNodeDebugString(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->DebugString();
}

void AnfAlgo::SetNodeAttr(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  // this function was moved to AnfUtils.
  return AnfUtils::SetNodeAttr(key, value, node);
}

void AnfAlgo::SetNodeAttrSafely(const std::string &key, const ValuePtr &value, const AnfNodePtr &node) {
  // Make CNode safe to set attr firstly.
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr) {
    return;
  }
  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  if (prim != nullptr) {
    auto new_prim = prim->isa<PrimitivePy>() ? prim : prim->Clone();
    cnode->set_input(0, NewValueNode(new_prim));
  }

  // Set attr secondly.
  common::AnfAlgo::SetNodeAttr(key, value, node);
}

void AnfAlgo::CopyNodeAttr(const std::string &key, const AnfNodePtr &from, const AnfNodePtr &to) {
  CopyNodeAttr(key, key, from, to);
}

void AnfAlgo::CopyNodeAttr(const std::string &old_key, const std::string &new_key, const AnfNodePtr &from,
                           const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (!from->isa<CNode>() || !to->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Only cnode has attr, but this from_anf is " << from->DebugString() << " ,to_node is "
                               << to->DebugString() << trace::DumpSourceLines(from);
  }
  auto from_primitive = AnfAlgo::GetCNodePrimitive(from);
  MS_EXCEPTION_IF_NULL(from_primitive);
  auto to_primitive = AnfAlgo::GetCNodePrimitive(to);
  MS_EXCEPTION_IF_NULL(to_primitive);
  to_primitive->set_attr(new_key, from_primitive->GetAttr(old_key));
}

void AnfAlgo::CopyNodeAttrs(const AnfNodePtr &from, const AnfNodePtr &to) {
  MS_EXCEPTION_IF_NULL(from);
  MS_EXCEPTION_IF_NULL(to);
  if (!from->isa<CNode>() || !to->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Only cnode has attr, but this from_anf is " << from->DebugString() << ",to_node is "
                               << from->DebugString() << trace::DumpSourceLines(from);
  }
  auto from_primitive = AnfAlgo::GetCNodePrimitive(from);
  MS_EXCEPTION_IF_NULL(from_primitive);
  auto to_primitive = AnfAlgo::GetCNodePrimitive(to);
  MS_EXCEPTION_IF_NULL(to_primitive);
  auto from_cnode = from->cast<CNodePtr>();
  auto to_cnode = to->cast<CNodePtr>();
  if (from_cnode->HasPrimalAttr(kAttrMicro)) {
    to_cnode->AddPrimalAttr(kAttrMicro, from_cnode->GetPrimalAttr(kAttrMicro));
  }
  (void)to_primitive->SetAttrs(from_primitive->attrs());
}

void AnfAlgo::EraseNodeAttr(const std::string &key, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Only cnode has attr, but this anf is " << node->DebugString()
                               << trace::DumpSourceLines(node);
  }
  // single op cnode.
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive != nullptr) {
    primitive->EraseAttr(key);
    return;
  }
  // graph kernel cnode.
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  fg->erase_flag(key);
}

bool AnfAlgo::HasNodeAttr(const std::string &key, const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // call node's input0 is not a primitive.
  if (!IsValueNode<FuncGraph>(node->input(0)) && !IsValueNode<Primitive>(node->input(0))) {
    return false;
  }
  // single op cnode.
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive != nullptr) {
    return primitive->HasAttr(key);
  }
  // graph kernel cnode.
  auto fg = AnfAlgo::GetCNodeFuncGraphPtr(node);
  MS_EXCEPTION_IF_NULL(fg);
  return fg->has_attr(key);
}

size_t AnfAlgo::GetInputNum(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  size_t input_num = cnode->size();
  if (input_num == 0) {
    MS_LOG(INTERNAL_EXCEPTION) << "Cnode inputs size can't be zero." << trace::DumpSourceLines(cnode);
  }
  return input_num - 1;
}

size_t AnfAlgo::GetInputTensorNum(const AnfNodePtr &node) {
  // this function was moved to AnfUtils.
  return AnfUtils::GetInputTensorNum(node);
}

bool AnfAlgo::IsPrevNodeHasTupleGetItem(const AnfNodePtr &anf_node, size_t input_idx, bool skip_nop_node) {
  if (!anf_node->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << anf_node->DebugString() << "anf_node is not CNode."
                               << trace::DumpSourceLines(anf_node);
  }
  auto input_node = AnfAlgo::GetInputNode(anf_node->cast<CNodePtr>(), input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  auto res = VisitKernelWithReturnType(input_node, 0, skip_nop_node, {prim::kPrimTupleGetItem});
  if (CheckPrimitiveType(res.first, prim::kPrimTupleGetItem)) {
    return true;
  }
  return false;
}

KernelWithIndex AnfAlgo::GetPrevNodeOutput(const AnfNodePtr &anf_node, size_t input_idx, bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (!anf_node->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << anf_node->DebugString() << "anf_node is not CNode."
                               << trace::DumpSourceLines(anf_node);
  }
  auto kernel_info = anf_node->kernel_info();
  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      auto output = runtime_cache.runtime_cache().get_prev_node_output(input_idx);
      if (output.first != nullptr) {
        return output;
      }
    }
  }
  KernelWithIndex res;
  if (CheckPrimitiveType(anf_node, prim::kPrimTupleGetItem)) {
    res = VisitKernelWithReturnType(anf_node, 0, skip_nop_node);
  } else {
    auto input_node = AnfAlgo::GetInputNode(anf_node->cast<CNodePtr>(), input_idx);
    MS_EXCEPTION_IF_NULL(input_node);
    res = VisitKernelWithReturnType(input_node, 0, skip_nop_node);
  }
  if (kernel_info) {
    auto runtime_cache = kernel_info->runtime_cache();
    if (runtime_cache.runtime_cache().is_valid()) {
      runtime_cache.runtime_cache().set_prev_node_output(input_idx, res);
    }
  }
  return res;
}

// if the prev_node is MakeTuple, get all the input_nodes recursively, else use the ori GetPrevNodeOutput function
std::vector<KernelWithIndex> AnfAlgo::GetRealPrevNodesOutput(const AnfNodePtr &anf_node, size_t input_idx,
                                                             bool skip_nop_node) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::vector<KernelWithIndex> res;
  auto input_node = AnfAlgo::GetInputNode(cnode, input_idx);
  MS_EXCEPTION_IF_NULL(input_node);
  if (CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
    auto maketuple_input_num = GetInputTensorNum(input_node);
    for (size_t i = 0; i < maketuple_input_num; ++i) {
      auto inputs_i = GetRealPrevNodesOutput(input_node, i, skip_nop_node);
      (void)res.insert(res.end(), inputs_i.begin(), inputs_i.end());
    }
  } else {
    (void)res.emplace_back(GetPrevNodeOutput(cnode, input_idx, skip_nop_node));
  }
  return res;
}

std::vector<TypeId> AnfAlgo::GetRealPrevNodesOutputInferDataType(const AnfNodePtr &node, size_t input_idx) {
  std::vector<KernelWithIndex> kernels_with_index = AnfAlgo::GetRealPrevNodesOutput(node, input_idx);
  std::vector<TypeId> res;
  (void)std::transform(kernels_with_index.begin(), kernels_with_index.end(), std::back_inserter(res),
                       [](auto kernel_with_index) {
                         return AnfAlgo::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
                       });
  return res;
}

namespace {
inline ShapeVector GetShape(const abstract::BaseShapePtr &base_shape) {
  MS_EXCEPTION_IF_NULL(base_shape);
  if (base_shape->isa<abstract::Shape>()) {
    auto shape_ptr = base_shape->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    return shape_ptr->shape();
  }
  return {};
}

ShapeVector GetOutputShape(const abstract::AbstractBasePtr &abstract, size_t output_idx, bool is_real_squence_output) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (abstract->isa<abstract::AbstractTensor>() || abstract->isa<abstract::AbstractMapTensor>()) {
    if (output_idx != 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "The abstract " << abstract->ToString()
                                 << "is single output but got index:" << output_idx;
    }
    const auto &shape = abstract->GetShape();
    return GetShape(shape);
  } else if (abstract->isa<abstract::AbstractScalar>() || abstract->isa<abstract::AbstractMonad>()) {
    return ShapeVector();
  } else if (abstract->isa<abstract::AbstractSparseTensor>()) {
    const auto &shape = abstract->GetShape();
    MS_EXCEPTION_IF_NULL(shape);
    const auto &tuple_shape = shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(tuple_shape);
    if (output_idx >= tuple_shape->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Output index " << output_idx << "is larger than output number "
                                 << tuple_shape->size() << " of tuple shape:" << tuple_shape->ToString()
                                 << " in abstract:" << abstract;
    }
    return GetShape(tuple_shape->shape()[output_idx]);
  }

  if (!abstract->isa<abstract::AbstractSequence>()) {
    MS_LOG(INFO) << "Unknown abstract for get shape:" << abstract->ToString();
    return {};
  }

  const auto &sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(sequence_abstract);
  if (sequence_abstract->dynamic_len()) {
    const auto &element_abstract = sequence_abstract->dynamic_len_element_abs();
    if (element_abstract == nullptr) {
      MS_LOG(ERROR) << "Invalid abstract for get shape:" << sequence_abstract->ToString();
      return ShapeVector();
    }
    return GetOutputShape(element_abstract, 0, true);
  }

  if (sequence_abstract->size() == 0) {
    return ShapeVector();
  }

  if (!is_real_squence_output) {
    if (output_idx >= sequence_abstract->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Output index " << output_idx << "is larger than output number "
                                 << sequence_abstract->size() << " of abstract:" << sequence_abstract->ToString();
    }
    MS_EXCEPTION_IF_NULL(sequence_abstract->elements()[output_idx]);
    return GetOutputShape(sequence_abstract->elements()[output_idx], 0, true);
  }

  // For real sequence output, if the inner elements' shape is same, the output is {element_num, *actual_shape},
  // otherwise is {element_num, inner_max_size}.
  // For example:
  //   1) Output abstract: ((3,4,5), (3,4,5)), output shape: (2, 3, 4, 5).
  //   2) Output abstract: ((3,4,5), (3,4,6)), output shape: (2, 72).
  ShapeVector elem_shape_vector;
  size_t change_cnt = 0;
  ShapeValueDType elem_size = 0;
  for (const auto &elem_abs : sequence_abstract->elements()) {
    MS_EXCEPTION_IF_NULL(elem_abs);
    elem_shape_vector = GetOutputShape(elem_abs, 0, true);
    auto cur_size = std::accumulate(elem_shape_vector.begin(), elem_shape_vector.end(), 1L, std::multiplies<int64_t>());
    if (elem_size < cur_size) {
      elem_size = cur_size;
      ++change_cnt;
    }
  }

  ShapeVector shape_vector = {SizeToLong(sequence_abstract->size())};
  if (change_cnt == 1) {
    (void)shape_vector.insert(shape_vector.end(), elem_shape_vector.begin(), elem_shape_vector.end());
  } else {
    shape_vector.push_back(elem_size);
  }
  return shape_vector;
}
}  // namespace

ShapeVector AnfAlgo::GetOutputInferShape(const AnfNodePtr &node, size_t output_idx, bool is_real_squence_output) {
  MS_EXCEPTION_IF_NULL(node);
  return GetOutputShape(node->abstract(), output_idx, is_real_squence_output || AnfAlgo::IsDynamicSequence(node));
}

ShapeVector AnfAlgo::GetPrevNodeOutputInferShape(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfAlgo::GetOutputInferShape(kernel_with_index.first, kernel_with_index.second);
}

TypePtr AnfAlgo::GetOutputInferType(const AnfNodePtr &node, size_t output_idx, bool is_real_tuple) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->abstract());
  const auto &type = node->abstract()->BuildType();
  MS_EXCEPTION_IF_NULL(type);
  if (!type->isa<Tuple>() && !type->isa<List>()) {
    if (output_idx != 0) {
      MS_LOG(EXCEPTION) << "Invalid index:" << output_idx << " for node:" << node->DebugString()
                        << " abstract:" << node->abstract()->ToString() << " type:" << type->ToString();
    }
    return type;
  }
  if (is_real_tuple) {
    return type;
  }
  if (type->isa<Tuple>()) {
    const auto &tuple_type = type->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_type);
    if (tuple_type->dynamic_len()) {
      if (output_idx != 0) {
        MS_LOG(EXCEPTION) << "Failed to get type by index:" << output_idx << " type:" << type->ToString();
      }
      return tuple_type;
    }
    if (output_idx >= tuple_type->size()) {
      MS_LOG(EXCEPTION) << "Invalid index:" << output_idx << " for node:" << node->DebugString()
                        << " abstract:" << node->abstract()->ToString() << " type:" << type->ToString();
    }
    return tuple_type->elements()[output_idx];
  }
  const auto &list_type = type->cast<ListPtr>();
  MS_EXCEPTION_IF_NULL(list_type);
  if (list_type->dynamic_len()) {
    if (output_idx != 0) {
      MS_LOG(EXCEPTION) << "Failed to get type by index:" << output_idx << " type:" << type->ToString();
    }
    return list_type;
  }
  if (output_idx >= list_type->size()) {
    MS_LOG(EXCEPTION) << "Invalid index:" << output_idx << " for node:" << node->DebugString()
                      << " abstract:" << node->abstract()->ToString() << " type:" << type->ToString();
  }
  return list_type->elements()[output_idx];
}

TypeId AnfAlgo::GetOutputInferDataType(const TypePtr &type, size_t output_idx) {
  auto type_ptr = type;
  MS_EXCEPTION_IF_NULL(type_ptr);
  if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    if (tuple_ptr->size() == 0) {
      if (tuple_ptr->dynamic_len() && tuple_ptr->dynamic_element_type() != nullptr) {
        MS_LOG(INFO) << "Dynamic empty tuple type has an dynamic element type:"
                     << tuple_ptr->dynamic_element_type()->type_id();
        return tuple_ptr->dynamic_element_type()->type_id();
      }
      return kTypeUnknown;
    }
    if (tuple_ptr->dynamic_len()) {
      MS_EXCEPTION_IF_NULL(tuple_ptr->dynamic_element_type());
      return GetOutputInferDataType(tuple_ptr->dynamic_element_type(), 0);
    }
    MS_EXCEPTION_IF_NULL(tuple_ptr);
    if (output_idx >= tuple_ptr->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Output index " << output_idx << " must be less than output number "
                                 << tuple_ptr->size();
    }
    type_ptr = (*tuple_ptr)[output_idx];
    MS_EXCEPTION_IF_NULL(type_ptr);
  }

  if (type_ptr->isa<List>()) {
    auto list_ptr = type_ptr->cast<ListPtr>();
    MS_EXCEPTION_IF_NULL(list_ptr);
    if (list_ptr->size() == 0) {
      if (list_ptr->dynamic_len() && list_ptr->dynamic_element_type() != nullptr) {
        MS_LOG(INFO) << "Dynamic empty list type has an dynamic element type:"
                     << list_ptr->dynamic_element_type()->type_id();
        return list_ptr->dynamic_element_type()->type_id();
      }
      return kTypeUnknown;
    }
    if (list_ptr->dynamic_len()) {
      MS_EXCEPTION_IF_NULL(list_ptr->dynamic_element_type());
      return GetOutputInferDataType(list_ptr->dynamic_element_type(), 0);
    }
    MS_EXCEPTION_IF_NULL(list_ptr);
    if (output_idx >= list_ptr->size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Output index " << output_idx << " must be less than output number "
                                 << list_ptr->size();
    }
    type_ptr = (*list_ptr)[output_idx];
    MS_EXCEPTION_IF_NULL(type_ptr);
  }

  if (type_ptr->isa<SparseTensorType>()) {
    auto tensor_ptr = type_ptr->cast<SparseTensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    type_ptr = (*tensor_ptr)[output_idx];
    MS_EXCEPTION_IF_NULL(type_ptr);
  }

  if (type_ptr->isa<TensorType>()) {
    auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    TypePtr elem = tensor_ptr->element();
    MS_EXCEPTION_IF_NULL(elem);
    return elem->type_id();
  }
  if (type_ptr->isa<Tuple>() || type_ptr->isa<List>()) {
    return GetOutputInferDataType(type_ptr, 0);
  }
  return type_ptr->type_id();
}

namespace {
bool IsTupleInTupleValueNode(const AnfNodePtr &node) {
  if (node == nullptr || !node->isa<ValueNode>()) {
    return false;
  }
  const auto &value_node = node->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  const auto &value = value_node->value();
  if (value == nullptr || !value->isa<ValueSequence>()) {
    return false;
  }
  const auto &value_sequence = value->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(value_sequence);
  return std::any_of(value_sequence->value().begin(), value_sequence->value().end(),
                     [](const ValuePtr &sub_value) { return sub_value != nullptr && sub_value->isa<ValueSequence>(); });
}
}  // namespace

TypeId AnfAlgo::GetOutputInferDataType(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsCallNode(node) || IsTupleInTupleValueNode(node)) {
    if (node->abstract() == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Empty abstract of call node:" << node->DebugString();
    }
    const auto &abs = common::AnfAlgo::FetchAbstractByIndex(node->abstract(), output_idx);
    MS_EXCEPTION_IF_NULL(abs);
    const auto &type = abs->BuildType();
    MS_EXCEPTION_IF_NULL(type);
    if (type->isa<TensorType>()) {
      const auto &tensor_type = type->cast<TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type);
      const auto &element = tensor_type->element();
      return element->type_id();
    } else {
      return type->type_id();
    }
  }
  return GetOutputInferDataType(node->Type(), output_idx);
}

TypeId AnfAlgo::GetPrevNodeOutputInferDataType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfAlgo::GetOutputInferDataType(kernel_with_index.first, kernel_with_index.second);
}

TypePtr AnfAlgo::GetPrevNodeOutputInferType(const AnfNodePtr &node, size_t input_idx) {
  KernelWithIndex kernel_with_index = AnfAlgo::GetPrevNodeOutput(node, input_idx);
  return AnfAlgo::GetOutputInferType(kernel_with_index.first, kernel_with_index.second);
}

// set infer shapes and types of anf node
void AnfAlgo::SetOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                          const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_ptr = node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node_ptr);
  std::string node_name = "";
  if (node_ptr->isa<CNode>()) {
    node_name = GetCNodeName(node_ptr);
  }
  if (types.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Types size " << types.size() << "should be same with shapes size " << shapes.size()
                               << " for node " << node->fullname_with_scope() << "." << trace::DumpSourceLines(node);
  }

  auto tuple_node = kNodeTupleOutSet.find(node_name);
  if (shapes.empty() && tuple_node == kNodeTupleOutSet.end()) {
    node->set_abstract(std::make_shared<abstract::AbstractNone>());
  } else if (shapes.size() == 1 && tuple_node == kNodeTupleOutSet.end()) {
    // single output handle
    if (shapes[0]->isa<abstract::NoShape>()) {
      auto abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(types[0]));
      node->set_abstract(abstract);
    } else {
      auto abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]), shapes[0]);
      node->set_abstract(abstract);
    }
  } else {
    // multiple output handle
    std::vector<AbstractBasePtr> abstract_list;
    for (size_t i = 0; i < types.size(); ++i) {
      if (shapes[0]->isa<abstract::NoShape>()) {
        auto abstract = std::make_shared<abstract::AbstractScalar>(TypeIdToType(types[i]));
        abstract_list.emplace_back(abstract);
      } else {
        auto abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[i]), shapes[i]);
        abstract_list.emplace_back(abstract);
      }
    }
    auto abstract_tuple = std::make_shared<AbstractTuple>(abstract_list);
    node->set_abstract(abstract_tuple);
  }
}

void AnfAlgo::SetSingleOutputTypeAndDetailShape(const std::vector<TypeId> &types,
                                                const std::vector<abstract::BaseShapePtr> &shapes, AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_ptr = node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node_ptr);
  if (types.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Types size " << types.size() << "should be same with shapes size " << shapes.size()
                               << " for node " << node->fullname_with_scope() << "." << trace::DumpSourceLines(node);
  }
  auto abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]), shapes[0]);
  node->set_abstract(abstract);
}

namespace {
void DeleteDynamicLen(AnfNode *node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->abstract() != nullptr && node->abstract()->isa<abstract::AbstractSequence>()) {
    const auto &tuple_abs = node->abstract()->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(tuple_abs);
    if (tuple_abs->dynamic_len()) {
      auto cloned_abstract = tuple_abs->Clone()->cast<abstract::AbstractSequencePtr>();
      cloned_abstract->set_dynamic_len(false);
      node->set_abstract(cloned_abstract);
    }
  }
}
}  // namespace

// set infer shapes and types of anf node
void AnfAlgo::SetOutputInferTypeAndShape(const std::vector<TypeId> &types, const std::vector<ShapeVector> &shapes,
                                         AnfNode *node, bool disable_dynamic_len) {
  MS_EXCEPTION_IF_NULL(node);
  if (disable_dynamic_len) {
    DeleteDynamicLen(node);
  }
  auto node_ptr = node->cast<AnfNodePtr>();
  MS_EXCEPTION_IF_NULL(node_ptr);
  std::string node_name = "";
  if (node_ptr->isa<CNode>()) {
    node_name = GetCNodeName(node_ptr);
  }
  if (types.size() != shapes.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Types size " << types.size() << "should be same with shapes size " << shapes.size()
                               << "." << trace::DumpSourceLines(node);
  }
  auto abstract_ptr = node_ptr->abstract();

  auto tuple_node = kNodeTupleOutSet.find(node_name);
  if (shapes.empty() && tuple_node == kNodeTupleOutSet.end()) {
    node->set_abstract(std::make_shared<abstract::AbstractNone>());
  } else if (shapes.size() == 1 && tuple_node == kNodeTupleOutSet.end()) {
    // single output handle
    if (abstract_ptr != nullptr && abstract_ptr->isa<abstract::AbstractMapTensor>()) {
      // For AbstractMapTensor.
      abstract_ptr->set_shape(std::make_shared<abstract::Shape>(shapes[0]));
      return;
    }

    abstract::AbstractTensorPtr abstract = std::make_shared<AbstractTensor>(TypeIdToType(types[0]), shapes[0]);
    node->set_abstract(abstract);
  } else {
    // multiple output handle
    std::vector<AbstractBasePtr> abstract_list;
    for (size_t i = 0; i < types.size(); ++i) {
      abstract::AbstractTensorPtr abstract =
        std::make_shared<AbstractTensor>(TypeIdToType(types[i]), std::make_shared<abstract::Shape>(shapes[i]));
      abstract_list.emplace_back(abstract);
    }
    auto abstract_tuple = std::make_shared<AbstractTuple>(abstract_list);
    node->set_abstract(abstract_tuple);
  }
}
// copy an abstract of a node to another node
void AnfAlgo::CopyAbstract(const AnfNodePtr &from_node, AnfNode *to_node) {
  MS_EXCEPTION_IF_NULL(from_node);
  MS_EXCEPTION_IF_NULL(to_node);
  to_node->set_abstract(from_node->abstract());
}

bool AnfAlgo::IsNodeInGraphKernel(const AnfNodePtr &node) {
  // this function was moved to AnfUtils.
  return AnfUtils::IsNodeInGraphKernel(node);
}

AnfNodePtr AnfAlgo::GetOutputOfGraphkernel(const KernelWithIndex &kernel_with_index) {
  auto func_graph = GetCNodeFuncGraph(kernel_with_index.first);
  if (func_graph == nullptr) {
    return kernel_with_index.first;
  }
  auto output = func_graph->output();
  if (CheckPrimitiveType(output, prim::kPrimMakeTuple)) {
    return output->cast<CNodePtr>()->input(kernel_with_index.second + 1);
  }
  return output;
}

bool AnfAlgo::IsParameterWeight(const ParameterPtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->has_default();
}

bool AnfAlgo::IsLabelIndexInNode(const AnfNodePtr &node, size_t label_index) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(cnode) == kLabelGotoOpName &&
      (AnfAlgo::GetNodeAttr<uint32_t>(cnode, kAttrLabelIndex) == label_index)) {
    return true;
  } else if (AnfAlgo::GetCNodeName(cnode) == kLabelSwitchOpName) {
    auto label_list = AnfAlgo::GetNodeAttr<std::vector<uint32_t>>(cnode, kAttrLabelSwitchList);
    if (std::find(label_list.begin(), label_list.end(), label_index) != label_list.end()) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::IsUpdateParameterKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_name = GetCNodeName(node);
  if (HasNodeAttr(kAttrAsync, node) && GetNodeAttr<bool>(node, kAttrAsync)) {
    return false;
  }
  if (!IsOneOfOperator(node_name) && node_name.find("Assign") == string::npos) {
    return false;
  }
  return true;
}

bool AnfAlgo::IsTupleOutput(const AnfNodePtr &anf) {
  MS_EXCEPTION_IF_NULL(anf);
  TypePtr type = anf->Type();
  if (type == nullptr) {
    return false;
  }

  // For dynamic sequence node, all output should be emplaced in single tensor.
  if (anf->abstract() && IsDynamicSequence(anf)) {
    return false;
  }

  MS_EXCEPTION_IF_NULL(type);
  return type->isa<Tuple>() || type->isa<List>() || type->isa<SparseTensorType>();
}

AnfNodePtr AnfAlgo::GetInputNode(const CNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto get_input_index = index + 1;
  if (get_input_index >= node->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Input index size " << get_input_index << ", but the node input size just "
                               << node->size() << ". node: " << node->DebugString() << "."
                               << trace::DumpSourceLines(node);
  }
  // input 0 is primitive node
  return node->input(get_input_index);
}

void AnfAlgo::SetNodeInput(const CNodePtr &node, const AnfNodePtr &input_node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(input_node);
  if (node->func_graph() != nullptr) {
    auto manager = node->func_graph()->manager();
    if (manager != nullptr) {
      manager->SetEdge(node, SizeToInt(index + 1), input_node);
      return;
    }
  }
  node->set_input(index + 1, input_node);
}

AnfNodePtr AnfAlgo::GetCNodePrimitiveNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return node->input(kAnfPrimitiveIndex);
}

PrimitivePtr AnfAlgo::GetCNodePrimitive(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto attr_input = GetCNodePrimitiveNode(cnode);
  MS_EXCEPTION_IF_NULL(attr_input);
  auto value_node = attr_input->cast<ValueNodePtr>();
  MS_EXCEPTION_IF_NULL(value_node);
  auto value = value_node->value();
  MS_EXCEPTION_IF_NULL(value);
  auto primitive = value->cast<PrimitivePtr>();
  return primitive;
}

bool AnfAlgo::IsInplaceNode(const mindspore::AnfNodePtr &kernel, const string &type) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto primitive = AnfAlgo::GetCNodePrimitive(kernel);
  if (!primitive) {
    return false;
  }

  auto inplace_attr = primitive->GetAttr(type);
  if (inplace_attr == nullptr) {
    return false;
  }

  return true;
}

bool AnfAlgo::IsCommunicationOp(const std::string &prim_name) {
  static const std::set<std::string> kCommunicationOpNames = {
    kAllReduceOpName,       kAllGatherOpName,       kBroadcastOpName, kReduceScatterOpName,     kSendOpName,
    kReceiveOpName,         kAlltoAllOpName,        kAllToAllOpName,  kAllToAllvOpName,         kMuxReceiveOpName,
    kMuxSendOpName,         kReduceOpName,          kBarrierOpName,   kCollectiveScatterOpName, kCollectiveGatherOpName,
    kMatMulAllReduceOpName, kBatchISendIRecvOpName, kAlltoAllVOpName};
  return (kCommunicationOpNames.find(prim_name) != kCommunicationOpNames.end());
}

bool AnfAlgo::IsCommunicationOp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  return IsCommunicationOp(kernel_name);
}

bool AnfAlgo::IsDtypeFormatSensitiveOp(const AnfNodePtr &node) {
  static const std::set<std::string> kDtypeFormatSensitiveOpNames = {kCastOpName};
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  return (kDtypeFormatSensitiveOpNames.find(kernel_name) != kDtypeFormatSensitiveOpNames.end());
}

bool AnfAlgo::IsFusedCommunicationOp(const AnfNodePtr &node) {
  if (!IsCommunicationOp(node)) {
    return false;
  }
  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  MS_EXCEPTION_IF_NULL(primitive);
  ValuePtr attr_fusion = primitive->GetAttr(kAttrFusion);
  ValuePtr attr_not_delay_fusion = primitive->GetAttr(kAttrNotDelayFusion);
  if (attr_fusion == nullptr) {
    return false;
  }

  auto fusion = GetValue<int64_t>(attr_fusion);
  if (fusion == 0) {
    return false;
  }
  if (attr_not_delay_fusion && GetValue<bool>(attr_not_delay_fusion)) {
    return false;
  }
  return true;
}

bool AnfAlgo::IsGetNext(const NotNull<AnfNodePtr> &node) {
  auto kernel_name = AnfAlgo::GetCNodeName(node);
  return kernel_name == kGetNextOpName || kernel_name == kDynamicGetNextV2OpName;
}

bool AnfAlgo::IsGraphKernel(const AnfNodePtr &node) {
  // this function was moved to AnfUtils.
  return AnfUtils::IsGraphKernel(node);
}

bool AnfAlgo::IsNeedSkipNopOpAddr(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive == nullptr) {
    return false;
  }

  auto skip_nop_op_addr_attr = primitive->GetAttr(kAttrSkipNopOpAddr);
  if (skip_nop_op_addr_attr == nullptr) {
    return false;
  }

  return GetValue<bool>(skip_nop_op_addr_attr);
}

bool AnfAlgo::IsNeedSkipNopOpExecution(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  auto primitive = AnfAlgo::GetCNodePrimitive(node);
  if (primitive == nullptr) {
    return false;
  }

  auto skip_nop_execution_attr = primitive->GetAttr(kAttrSkipNopOpExecution);
  if (skip_nop_execution_attr == nullptr) {
    return false;
  }

  return GetValue<bool>(skip_nop_execution_attr);
}

FuncGraphPtr AnfAlgo::GetValueNodeFuncGraph(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto value_node = node->cast<ValueNodePtr>();
  if (value_node == nullptr) {
    return nullptr;
  }
  auto value = value_node->value();
  if (value == nullptr) {
    return nullptr;
  }
  auto func_graph = value->cast<FuncGraphPtr>();
  return func_graph;
}

bool AnfAlgo::IsSwitchCall(const CNodePtr &call_node) {
  MS_EXCEPTION_IF_NULL(call_node);
  if (!CheckPrimitiveType(call_node, prim::kPrimCall)) {
    MS_LOG(INTERNAL_EXCEPTION) << "Call node should be a 'call', but is a " << call_node->DebugString() << "."
                               << trace::DumpSourceLines(call_node);
  }
  auto input1 = call_node->input(1);
  MS_EXCEPTION_IF_NULL(input1);
  if (input1->isa<ValueNode>()) {
    return false;
  } else if (input1->isa<CNode>() && AnfAlgo::CheckPrimitiveType(input1, prim::kPrimSwitch)) {
    return true;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Unexpected input1 of call node,input1:" << input1->DebugString() << "."
                             << trace::DumpSourceLines(call_node);
}

bool AnfAlgo::IsScalarInput(const CNodePtr &cnode, size_t index) {
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape.empty()) {
    return true;
  }
  return shape.size() == kShape1dDims && shape[0] == 1;
}

bool AnfAlgo::IsScalarOutput(const CNodePtr &cnode, size_t index) {
  auto shape = AnfAlgo::GetPrevNodeOutputInferShape(cnode, index);
  if (shape.empty()) {
    return true;
  }
  return shape.size() == kShape1dDims && shape[0] == 1;
}

namespace {
void FindDelayExecPosition(const std::vector<CNodePtr> &nodes, size_t current_index, std::set<size_t> *invalid_position,
                           std::map<size_t, std::vector<CNodePtr>> *insert_nodes) {
  MS_EXCEPTION_IF_NULL(invalid_position);
  MS_EXCEPTION_IF_NULL(insert_nodes);
  if (current_index >= nodes.size()) {
    return;
  }
  auto &node = nodes[current_index];
  for (size_t j = current_index + 1; j < nodes.size(); ++j) {
    auto &child = nodes[j];
    auto child_name = AnfAlgo::GetCNodeName(child);
    if (child_name == kAssignAddOpName || child_name == kAssignSubOpName || child_name == kAssignOpName ||
        IsOneOfOperator(child_name)) {
      return;
    }

    auto input_size = child->size() - 1;
    for (size_t k = 0; k < input_size; ++k) {
      auto kernel_index = AnfAlgo::GetPrevNodeOutput(child, k, true);
      if (kernel_index.first != node) {
        continue;
      }
      (void)invalid_position->insert(current_index);
      auto iter = insert_nodes->find(j);
      if (iter != insert_nodes->end()) {
        iter->second.emplace_back(node);
      } else {
        (*insert_nodes)[j] = {node};
      }
      return;
    }
  }
}

std::vector<CNodePtr> DelayExecNode(const std::vector<CNodePtr> &nodes, const std::string &node_name, bool only_seed) {
  std::map<size_t, std::vector<CNodePtr>> insert_nodes;
  std::set<size_t> invalid_position;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto &node = nodes[i];
    if (AnfAlgo::GetCNodeName(node) != node_name) {
      continue;
    }
    if (only_seed) {
      bool is_seed = true;
      auto input_size = node->size() - 1;
      for (size_t k = 0; k < input_size; ++k) {
        auto input = AnfAlgo::GetPrevNodeOutput(node, k, true).first;
        if (input != nullptr && input->isa<CNode>()) {
          is_seed = false;
          break;
        }
      }
      if (!is_seed) {
        continue;
      }
    }
    FindDelayExecPosition(nodes, i, &invalid_position, &insert_nodes);
  }
  std::vector<CNodePtr> result;
  for (size_t i = 0; i < nodes.size(); ++i) {
    auto iter = insert_nodes.find(i);
    if (iter != insert_nodes.end()) {
      (void)result.insert(result.end(), iter->second.rbegin(), iter->second.rend());
    }
    if (invalid_position.find(i) != invalid_position.end()) {
      continue;
    }
    result.emplace_back(nodes[i]);
  }
  return result;
}
}  // namespace

void AnfAlgo::ReorderExecList(NotNull<std::vector<CNodePtr> *> node_list) {
  std::vector<CNodePtr> result;
  std::copy(node_list->begin(), node_list->end(), std::back_inserter(result));
  result = DelayExecNode(result, kTransDataOpName, true);
  result = DelayExecNode(result, kCastOpName, true);
  result = DelayExecNode(result, kAdamApplyOneWithDecayOpName, false);
  result = DelayExecNode(result, kAdamApplyOneOpName, false);
  result = DelayExecNode(result, kQuantDTypeCastOpName, false);
  result = DelayExecNode(result, kFSEDecodeOpName, false);
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    result = DelayExecNode(result, kDropoutGenMaskOpName, true);
    result = DelayExecNode(result, kStatelessDropOutGenMaskOpName, true);
  }
  node_list->clear();
  std::copy(result.begin(), result.end(), std::back_inserter(*node_list));
}

void AnfAlgo::ReorderPosteriorExecList(NotNull<std::vector<CNodePtr> *> node_list) {
  std::vector<CNodePtr> ordinary_node_list;
  std::vector<CNodePtr> posterior_node_list;

  for (const auto &node : *node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (IsOneOfPosteriorOperator(AnfAlgo::GetCNodeName(node))) {
      posterior_node_list.emplace_back(node);
    } else {
      ordinary_node_list.emplace_back(node);
    }
  }
  node_list->clear();
  std::copy(ordinary_node_list.begin(), ordinary_node_list.end(), std::back_inserter(*node_list));
  std::copy(posterior_node_list.begin(), posterior_node_list.end(), std::back_inserter(*node_list));
}

TypeId AnfAlgo::GetCNodeOutputPrecision(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto prim = AnfAlgo::GetCNodePrimitive(node);
  if (prim == nullptr) {
    return kTypeUnknown;
  }

  TypeId except_type = kTypeUnknown;
  if (prim->GetAttr(kAttrOutputPrecision) != nullptr) {
    auto output_type_str = GetValue<std::string>(prim->GetAttr(kAttrOutputPrecision));
    if (output_type_str == "float16") {
      except_type = kNumberTypeFloat16;
    } else if (output_type_str == "float32") {
      except_type = kNumberTypeFloat32;
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "The fix precision must be float16 or float32, but got " << output_type_str << "."
                                 << trace::DumpSourceLines(node);
    }
  }

  return except_type;
}

TypeId AnfAlgo::GetPrevNodeOutputPrecision(const AnfNodePtr &node, size_t input_idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << node->DebugString() << ", input node is not CNode." << trace::DumpSourceLines(node);
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Input index " << input_idx << " is larger than input number "
                               << GetInputTensorNum(cnode) << "." << trace::DumpSourceLines(node);
  }
  auto input_node = cnode->input(input_idx + 1);
  MS_EXCEPTION_IF_NULL(input_node);
  auto kernel_with_index = VisitKernel(input_node, 0);
  if (!kernel_with_index.first->isa<CNode>()) {
    return kTypeUnknown;
  }
  return GetCNodeOutputPrecision(kernel_with_index.first);
}

bool AnfAlgo::IsCondControlKernel(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->inputs().empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Illegal null input of cnode." << trace::DumpSourceLines(node);
  }
  auto input = node->input(kAnfPrimitiveIndex);
  return IsPrimitive(input, prim::kPrimLabelGoto) || IsPrimitive(input, prim::kPrimLabelSwitch);
}

bool AnfAlgo::GetBooleanAttr(const AnfNodePtr &node, const std::string &attr) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto has_attr = AnfAlgo::HasNodeAttr(attr, cnode);
  if (!has_attr) {
    return false;
  }
  return AnfAlgo::GetNodeAttr<bool>(node, attr);
}

std::optional<string> AnfAlgo::GetDumpFlag(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  if (cnode == nullptr || !AnfAlgo::HasNodeAttr(kAttrDump, cnode)) {
    return {};
  }
  return std::optional<string>{AnfAlgo::GetNodeAttr<string>(node, kAttrDump)};
}

bool IsNodeDynamicRank(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Node is not a cnode";
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto in_dyn_rank = AnfAlgo::IsNodeInputDynamicRank(cnode);
  auto out_dyn_rank = AnfAlgo::IsNodeOutputDynamicRank(cnode);
  if (in_dyn_rank && !AnfAlgo::HasNodeAttr(kAttrInputIsDynamicRank, cnode)) {
    AnfAlgo::SetNodeAttrSafely(kAttrInputIsDynamicRank, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Set input dynamic rank attr for node:" << cnode->fullname_with_scope();
  }
  if (out_dyn_rank && !AnfAlgo::HasNodeAttr(kAttrOutputIsDynamicRank, cnode)) {
    AnfAlgo::SetNodeAttrSafely(kAttrOutputIsDynamicRank, MakeValue(true), cnode);
    MS_LOG(DEBUG) << "Set output dynamic rank attr for node:" << cnode->fullname_with_scope();
  }
  return in_dyn_rank || out_dyn_rank;
}

bool AnfAlgo::IsDynamicRankNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<Parameter>()) {
    return IsOutputAnchorDynamicRank(node, 0);
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if ((!HasNodeAttr(kAttrInputIsDynamicRank, cnode)) && (!HasNodeAttr(kAttrOutputIsDynamicRank, cnode))) {
    auto ret = IsNodeDynamicRank(node);
    MS_LOG(DEBUG) << "The Node:" << node->fullname_with_scope() << " is dynamic rank: [" << ret << "]";
    return ret;
  }
  return GetBooleanAttr(node, kAttrInputIsDynamicRank) || GetBooleanAttr(node, kAttrOutputIsDynamicRank) ||
         GetBooleanAttr(node, kAttrIsDynamicRank);
}

bool AnfAlgo::IsInputAnchorDynamicRank(const AnfNodePtr &node, size_t idx) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Only cnode has inputs, node: " << node->fullname_with_scope();
  }
  const auto &in_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, idx);
  if (mindspore::IsDynamicRank(in_shape)) {
    return true;
  }
  return false;
}

bool AnfAlgo::IsOutputAnchorDynamicRank(const AnfNodePtr &node, size_t idx) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &out_shape = common::AnfAlgo::GetOutputInferShape(node, idx);
  if (mindspore::IsDynamicRank(out_shape)) {
    return true;
  }
  return false;
}

bool AnfAlgo::IsNodeInputDynamicRank(const CNodePtr &anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  const auto &inputs = anf_node_ptr->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    if (IsNodeOutputDynamicRank(input)) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::IsNodeOutputDynamicRank(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    auto b_ptr = base_shape->cast<abstract::DynamicSequenceShapePtr>();
    if (b_ptr->IsDimUnknown()) {
      return true;
    }
  }
  return base_shape->IsDimUnknown();
}

bool AnfAlgo::IsDynamicShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Node is not a cnode.";
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if ((!HasNodeAttr(kAttrInputIsDynamicShape, cnode)) && (!HasNodeAttr(kAttrOutputIsDynamicShape, cnode))) {
    auto ret = IsNodeDynamicShape(node);
    MS_LOG(DEBUG) << "The Node:" << node->fullname_with_scope() << " is dynamic shape or not:" << ret;
    return ret;
  }
  return GetBooleanAttr(node, kAttrInputIsDynamicShape) || GetBooleanAttr(node, kAttrOutputIsDynamicShape);
}

bool AnfAlgo::IsDynamicValue(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    MS_LOG(DEBUG) << "Node is not a cnode.";
    return false;
  }
  if (AnfAlgo::IsGraphKernel(node)) {
    MS_LOG(DEBUG) << "Node(" << node->fullname_with_scope() << ") is GraphKernel node, it's not dynamic value type.";
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  if (cnode->HasAttr(ops::kHasDynamicValue)) {
    return true;
  }
  auto depend_list = abstract::GetValueDependArgIndices(cnode);
  if (!depend_list.empty()) {
    size_t real_input_num = cnode->size() - 1;  // exclude primitive in input[0]
    for (auto i = depend_list.begin(); i != depend_list.end(); i++) {
      if (*i >= SizeToInt(real_input_num)) {
        continue;
      }
      if (!cnode->input(*i + 1)->isa<ValueNode>()) {
        cnode->AddAttr(mindspore::ops::kHasDynamicValue, MakeValue(true));
        MS_LOG(DEBUG) << "The input index[" << *i << "]"
                      << " of node: " << cnode->fullname_with_scope() << " is a dynamic value input";
        return true;
      }
    }
  }
  return false;
}

void AnfAlgo::GetRealDynamicShape(const std::vector<size_t> &shape, NotNull<std::vector<int64_t> *> dynamic_shape) {
  for (auto size : shape) {
    if (size == SIZE_MAX) {
      dynamic_shape->push_back(-1);
    } else {
      dynamic_shape->push_back(SizeToLong(size));
    }
  }
}

static ShapeVector GetShapeFromSequenceShape(const abstract::SequenceShapePtr &sequeue_shape_ptr, size_t index) {
  MS_EXCEPTION_IF_NULL(sequeue_shape_ptr);
  auto shape_list = sequeue_shape_ptr->shape();
  if (index >= shape_list.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Output Index:" << index << " >= " << shape_list.size();
  }

  auto shape = shape_list[index];
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::NoShape>()) {
    // For scalar in sequeue case.
    return {};
  } else if (!shape->isa<abstract::Shape>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid Shape Type(" << shape->ToString() << ") In Shape List";
  }

  auto shape_ptr = shape->cast<abstract::ShapePtr>();
  return shape_ptr->max_shape();
}

ShapeVector AnfAlgo::GetOutputMaxShape(const AnfNodePtr &anf_node, size_t index) {
  MS_EXCEPTION_IF_NULL(anf_node);
  auto shape = anf_node->Shape();
  MS_EXCEPTION_IF_NULL(shape);
  if (shape->isa<abstract::Shape>()) {
    auto shape_ptr = shape->cast<abstract::ShapePtr>();
    return shape_ptr->max_shape();
  } else if (shape->isa<abstract::SequenceShape>()) {
    auto sequeue_shape_ptr = shape->cast<abstract::SequenceShapePtr>();
    return GetShapeFromSequenceShape(sequeue_shape_ptr, index);
  } else if (shape->isa<abstract::NoShape>()) {
    return {};
  } else if (shape->isa<abstract::DynamicSequenceShape>()) {
    return {1};
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid shape type." << trace::DumpSourceLines(anf_node);
  }
}

bool AnfAlgo::IsNodeOutputDynamicShape(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto base_shape = node->Shape();
  if (base_shape == nullptr) {
    MS_LOG(INFO) << "Invalid base shape, node: " << node->fullname_with_scope();
    return false;
  }
  if (base_shape->isa<abstract::DynamicSequenceShape>()) {
    return true;
  }
  return base_shape->IsDynamic();
}

bool AnfAlgo::IsNodeInputDynamicShape(const CNodePtr &anf_node_ptr) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  const auto &inputs = anf_node_ptr->inputs();
  for (size_t i = 1; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    if (IsNodeOutputDynamicShape(input)) {
      return true;
    }
  }
  return false;
}

std::string AnfAlgo::GetGraphSplitGroup(const AnfNodePtr &node) {
  return HasNodeAttr(kAttrGraphSplitGroup, node->cast<CNodePtr>())
           ? GetNodeAttr<std::string>(node->cast<CNodePtr>(), kAttrGraphSplitGroup)
           : "DefaultGroup";
}

void AnfAlgo::GetAllVisitedCNode(const CNodePtr &node, std::vector<AnfNodePtr> *used_kernels,
                                 std::set<AnfNodePtr> *visited) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(used_kernels);
  MS_EXCEPTION_IF_NULL(visited);
  if (visited->find(node) != visited->end()) {
    MS_LOG(INFO) << "Node:" << node->fullname_with_scope() << " has already been visited";
    return;
  }
  (void)visited->insert(node);
  auto input_size = node->size() - 1;
  for (size_t i = 0; i < input_size; ++i) {
    auto input = AnfAlgo::GetInputNode(node, i);
    if (!input->isa<CNode>()) {
      continue;
    }
    if (!AnfUtils::IsRealKernel(input) || IsNopNode(input)) {
      GetAllVisitedCNode(input->cast<CNodePtr>(), used_kernels, visited);
    } else {
      used_kernels->push_back(input);
    }
  }
}

void AnfAlgo::GetAllFatherRealNode(const AnfNodePtr &anf_node, std::vector<AnfNodePtr> *result,
                                   std::set<AnfNodePtr> *visited) {
  MS_EXCEPTION_IF_NULL(anf_node);
  MS_EXCEPTION_IF_NULL(result);
  MS_EXCEPTION_IF_NULL(visited);
  if (visited->find(anf_node) != visited->end()) {
    MS_LOG(INFO) << "Node:" << anf_node->fullname_with_scope() << " has already been visited";
    return;
  }
  visited->insert(anf_node);
  if (AnfUtils::IsRealKernel(anf_node)) {
    result->emplace_back(anf_node);
    return;
  }
  if (!anf_node->isa<CNode>()) {
    return;
  }
  auto cnode = anf_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Illegal null input of cnode(%s)" << anf_node->DebugString() << "."
                               << trace::DumpSourceLines(cnode);
  }
  auto input0 = cnode->input(0);
  if (IsPrimitive(input0, prim::kPrimMakeTuple)) {
    for (size_t i = 1; i < cnode->size(); ++i) {
      GetAllFatherRealNode(cnode->input(i), result, visited);
    }
  } else if (IsPrimitive(input0, prim::kPrimTupleGetItem)) {
    if (cnode->size() != kTupleGetItemInputSize) {
      MS_LOG(INTERNAL_EXCEPTION) << "The node tuple_get_item must have 2 inputs!";
    }
    GetAllFatherRealNode(cnode->input(kRealInputNodeIndexInTupleGetItem), result, visited);
  } else if (IsPrimitive(input0, prim::kPrimDepend)) {
    if (cnode->size() != kDependInputSize) {
      MS_LOG(INTERNAL_EXCEPTION) << "Depend node must have 2 inputs!" << trace::DumpSourceLines(cnode);
    }
    GetAllFatherRealNode(cnode->input(kRealInputIndexInDepend), result, visited);
    GetAllFatherRealNode(cnode->input(kDependAttachNodeIndex), result, visited);
  }
}

bool AnfAlgo::IsHostKernel(const CNodePtr &kernel_node) {
  static const std::map<std::string, std::pair<size_t, size_t>> host_kernel_input_output_num = {
    {prim::kPrimDynamicShape->name(), {1, 1}},
    {prim::kPrimReshape->name(), {2, 1}},
    {prim::kPrimTensorShape->name(), {1, 1}}};

  auto op_name = AnfAlgo::GetCNodeName(kernel_node);
  auto iter = host_kernel_input_output_num.find(op_name);
  if (iter == host_kernel_input_output_num.end()) {
    return false;
  }

  auto input_num = GetInputTensorNum(kernel_node);
  auto output_num = AnfUtils::GetOutputTensorNum(kernel_node);
  auto kernel_input_num = iter->second.first;
  auto kernel_output_num = iter->second.second;
  if (kernel_input_num != input_num || kernel_output_num != output_num) {
    return false;
  }
  return true;
}

void AnfAlgo::AddArgList(AbstractBasePtrList *args_spec_list, const AnfNodePtr &real_input, size_t real_input_index) {
  MS_EXCEPTION_IF_NULL(args_spec_list);
  MS_EXCEPTION_IF_NULL(real_input);

  // cppcheck-suppress unreadVariable
  auto lock = AnfUtils::GetAbstractLock(real_input.get());
  auto real_abs = real_input->abstract();
  MS_EXCEPTION_IF_NULL(real_abs);
  if (real_abs->isa<abstract::AbstractTuple>() && (!common::AnfAlgo::IsDynamicSequence(real_input))) {
    auto abs_tuple = real_abs->Clone()->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(abs_tuple);
    MS_EXCEPTION_IF_CHECK_FAIL((real_input_index < abs_tuple->elements().size()), "Index is out of range.");
    auto abs_index = abs_tuple->elements()[real_input_index];
    (void)args_spec_list->emplace_back(abs_index);
  } else {
    (void)args_spec_list->emplace_back(real_abs->Clone());
  }
}

AnfNodeIndexSet AnfAlgo::GetUpdateStateUsers(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  AnfNodeIndexSet update_states;
  for (auto &user : manager->node_users()[node]) {
    if (AnfAlgo::CheckPrimitiveType(user.first, prim::kPrimUpdateState)) {
      update_states.insert(user);
    }
  }
  return update_states;
}

void AnfAlgo::GetRealInputs(const AnfNodePtr &node, std::vector<KernelWithIndex> *inputs) {
  size_t input_num = AnfAlgo::GetInputTensorNum(node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    auto input_node = AnfAlgo::GetInputNode(node->cast<CNodePtr>(), input_index);
    GetRealOutputRecursively(input_node, 0, inputs);
  }
}

bool AnfAlgo::IsBpropCutOpExecInBackend(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  // Operators in set control_ops_exec_in_backend will be compiled into kernel graph, rather than be cut into single op
  // and executed in VM.
  static std::set<std::string> bprop_cut_ops_exec_in_backend = {kBpropCutOpName};
  return bprop_cut_ops_exec_in_backend.find(AnfAlgo::GetCNodeName(node)) != bprop_cut_ops_exec_in_backend.end();
}

bool AnfAlgo::IsNodeInputContainMonad(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto input_size = GetInputTensorNum(node);
  for (size_t i = 0; i < input_size; ++i) {
    auto input_with_index = GetPrevNodeOutput(node, i);
    if (HasAbstractMonad(input_with_index.first)) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::HasMonadInput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  const auto &inputs = cnode->inputs();
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (HasAbstractMonad(input)) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::IsNonTaskOp(const CNodePtr &node) {
  auto op_name = GetCNodeName(node);
  return (op_name == kSplitOpName || op_name == kSplitDOpName || op_name == kSplitVDOpName) &&
         AnfAlgo::HasNodeAttr(kAttrNonTask, node);
}

bool AnfAlgo::IsNoneInput(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, index);
  auto prev_node = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(prev_node);
  // Only const optional input(None) support now.
  if (prev_node->isa<ValueNode>()) {
    auto value = prev_node->cast<ValueNodePtr>()->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<None>()) {
      return true;
    }
  }

  return false;
}

bool AnfAlgo::IsCallNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto input0 = node->cast<CNodePtr>()->input(0);
  if (IsValueNode<Primitive>(input0)) {
    return false;
  }
  return true;
}

int64_t AnfAlgo::GetAttrGroups(const AnfNodePtr &node, size_t index) {
  if (node == nullptr) {
    return 1;
  }
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    if (HasNodeAttr(kAttrFracZGroupIdx, cnode)) {
      auto fz_group_idx = GetNodeAttr<std::vector<int64_t>>(cnode, kAttrFracZGroupIdx);
      if (index >= fz_group_idx.size()) {
        MS_LOG(INTERNAL_EXCEPTION) << "Index out of range, attr fracz_group_idx of node[" << node->fullname_with_scope()
                                   << "] only have " << fz_group_idx.size() << " numbers, but get index " << index;
      }
      return fz_group_idx[index];
    } else if (HasNodeAttr(kAttrFracZGroup, cnode)) {
      return GetNodeAttr<int64_t>(cnode, kAttrFracZGroup);
    }
  }
  if (node->isa<Parameter>()) {
    auto param = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
    return param->fracz_group();
  }
  if (node->isa<ValueNode>()) {
    auto value_node = node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    return value_node->fracz_group();
  }
  return 1;
}

AnfNodePtr AnfAlgo::GetTupleIndexes(const AnfNodePtr &node, std::vector<size_t> *const index_stack) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(index_stack);

  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto tuple_getitem = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getitem);
    // Get cur index
    auto output_index_value_node = tuple_getitem->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(output_index_value_node);
    auto value_node = output_index_value_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    auto output_idx = LongToSize(GetValue<int64_t>(value_node->value()));
    index_stack->push_back(output_idx);
    auto real_input = tuple_getitem->input(kRealInputNodeIndexInTupleGetItem);
    return GetTupleIndexes(real_input, index_stack);
  }
  if (IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
    // If make_tuple in make_tuple, visit may start with inner tuple_getitem.
    if (index_stack->empty()) {
      MS_LOG(WARNING) << "Visit make tuple: " << node->DebugString()
                      << ", but index are empty, visit should not start with inner tuple_getitem.";
      return nullptr;
    }
    auto make_tuple = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(make_tuple);
    auto output_idx = index_stack->back();
    index_stack->pop_back();
    return GetTupleIndexes(make_tuple->input(1 + output_idx), index_stack);
  }
  if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
    return GetTupleIndexes(node->cast<CNodePtr>()->input(kRealInputIndexInDepend), index_stack);
  }
  if (IsPrimitiveCNode(node, prim::kPrimLoad)) {
    return GetTupleIndexes(node->cast<CNodePtr>()->input(1), index_stack);
  }
  MS_LOG(DEBUG) << "Get real node:" << node->DebugString();
  return node;
}

bool AnfAlgo::IsNopNode(const AnfNodePtr &node) {
  static mindspore::HashSet<std::string> nop_nodes = {prim::kPrimReshape->name(),
                                                      kExpandDimsOpName,
                                                      prim::kPrimSqueeze->name(),
                                                      prim::kPrimFlatten->name(),
                                                      kFlattenGradOpName,
                                                      prim::kPrimReformat->name(),
                                                      prim::kPrimTupleToList->name(),
                                                      prim::kPrimListToTuple->name(),
                                                      prim::kPrimTupleToTensor->name(),
                                                      prim::kPrimScalarToTensor->name(),
                                                      prim::kPrimTensorToTuple->name(),
                                                      prim::kPrimTensorToScalar->name(),
                                                      "ReshapeExt"};
  if (node == nullptr || !node->isa<CNode>()) {
    return false;
  }
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().empty()) {
    return false;
  }
  auto input0 = cnode->input(0);
  MS_EXCEPTION_IF_NULL(input0);
  if (!input0->isa<ValueNode>()) {
    return false;
  }
  bool is_nop_node = false;
  if (AnfAlgo::HasNodeAttr(kAttrNopOp, cnode)) {
    is_nop_node = AnfAlgo::GetNodeAttr<bool>(cnode, kAttrNopOp);
  }
  if (nop_nodes.find(AnfAlgo::GetCNodeName(cnode)) == nop_nodes.end() && !is_nop_node) {
    return false;
  }

  // Check the input type and output type.
  if (GetOutputInferDataType(node, 0) != GetPrevNodeOutputInferDataType(node, 0)) {
    return false;
  }

  return true;
}

template <typename T>
bool AnfAlgo::CheckAbsType(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->abstract());
  return node->abstract()->cast<T>() != nullptr;
}

bool AnfAlgo::CheckAbsSparseTensor(const AnfNodePtr &node) {
  return CheckAbsType<abstract::AbstractSparseTensorPtr>(node);
}

bool AnfAlgo::CheckAbsSparseTensor(const abstract::AbstractBasePtr &abs) {
  return abs->cast<abstract::AbstractSparseTensorPtr>() != nullptr;
}

TypeId AnfAlgo::GetSparseTypeIdAt(const AnfNodePtr &node, size_t idx) {
  if (CheckAbsType<abstract::AbstractSparseTensorPtr>(node)) {
    auto abs_sparse = node->abstract()->cast<abstract::AbstractSparseTensorPtr>();
    auto shape_idx = abs_sparse->size() - 1;
    // idx points to a tensor element
    if (idx < shape_idx) {
      return abs_sparse->GetTensorTypeIdAt(idx);
    }
    return abs_sparse->GetShapeTypeIdAt(idx - shape_idx);
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Expect AbstractCSRTensor or AbstractCOOTensor, but got "
                             << node->abstract()->ToString();
}

std::string AnfAlgo::GetTensorValueString(const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto dtype = tensor->Dtype();
  MS_EXCEPTION_IF_NULL(dtype);
  size_t data_size = tensor->DataSize();
  auto shape = tensor->shape();
  std::ostringstream buf;
  auto fn = [&buf, data_size, &shape](auto addr) {
    // Tensor value.
    buf << "v";
    for (size_t i = 0; i < data_size; ++i) {
      buf << *(addr + i) << ",";
    }
    // Tensor shape is necessary.
    // For example, the value of ones[3x4] and ones[4x3] are the same, but the shape is different.
    buf << "s" << tensor::ShapeToString(shape);
  };

  if (dtype->type_id() == kNumberTypeBool) {
    fn(reinterpret_cast<bool *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeInt) {
    fn(reinterpret_cast<int *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeInt8) {
    fn(reinterpret_cast<int8_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeUInt8) {
    fn(reinterpret_cast<uint8_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeInt16) {
    fn(reinterpret_cast<int16_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeUInt16) {
    fn(reinterpret_cast<uint16_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeInt32) {
    fn(reinterpret_cast<int32_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeUInt32) {
    fn(reinterpret_cast<uint32_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeInt64) {
    fn(reinterpret_cast<int64_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeUInt64) {
    fn(reinterpret_cast<uint64_t *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeFloat16) {
    fn(reinterpret_cast<float16 *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeFloat64) {
    fn(reinterpret_cast<double *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeFloat || dtype->type_id() == kNumberTypeFloat32) {
    fn(reinterpret_cast<float *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeBFloat16) {
    fn(reinterpret_cast<bfloat16 *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeComplex64) {
    fn(reinterpret_cast<complex64 *>(tensor->data_c()));
  } else if (dtype->type_id() == kNumberTypeComplex128) {
    fn(reinterpret_cast<complex128 *>(tensor->data_c()));
  } else {
    MS_LOG(INTERNAL_EXCEPTION) << "The dtype of the constant input is " << dtype->ToString();
  }
  return buf.str();
}

abstract::AbstractBasePtr AnfAlgo::FrontendGetNodeAbstractByIndex(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &abstract = node->abstract();
  if (abstract == nullptr) {
    return abstract;
  }

  // Return output abstract directly for : 1.not sequence type, 2.dynamic sequence type, 3.real tuple/list type.
  if (!abstract->isa<abstract::AbstractSequence>() || common::AnfAlgo::IsDynamicSequence(node)) {
    MS_EXCEPTION_IF_CHECK_FAIL((index == 0),
                               "Cannot get " + std::to_string(index) + " child abstract from " + abstract->ToString());
    return abstract;
  }

  // Return element abstract by index for tuple type.
  const auto &abstract_tuple = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(abstract_tuple);
  const auto &elements = abstract_tuple->elements();
  if (elements.size() <= index) {
    const auto sub_abstract = FetchAbstractByIndex(node->abstract(), index);
    return sub_abstract;
  }
  return elements[index];
}

std::string AnfAlgo::GetJitLevel(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  if (!func_graph->has_attr(kAttrJitLevel)) {
    MS_LOG(INFO) << "The func_graph:" << func_graph->ToString() << " has no jit_level attr, return default: None.";
    return "";
  }
  auto jit_level_value = func_graph->get_attr(kAttrJitLevel);
  auto jit_level = GetValue<std::string>(jit_level_value);
  return jit_level;
}

bool AnfAlgo::IsNodeMutableScalar(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  // Check if the node is mutable scalar by all_inputs are scalar or output is scalar.
  const auto &is_mutable_scalar_func = [](const AnfNodePtr &cur_node) {
    const auto &abstract = cur_node->abstract();
    if (abstract == nullptr || (!abstract->isa<abstract::AbstractScalar>())) {
      return false;
    }
    if (abstract->BuildValue()->ContainsValueAny() && abstract->BuildType()->isa<Number>()) {
      return true;
    }
    return false;
  };
  bool is_output_mutable_scalar = is_mutable_scalar_func(node);
  bool is_scalar_to_tensor = IsPrimitiveCNode(node, prim::kPrimScalarToTensor);
  if (AnfAlgo::CheckPrimitiveType(node, prim::kPrimDepend)) {
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (!is_mutable_scalar_func(cnode->input(kRealInputIndexInDepend))) {
      return false;
    }
  }
  return is_output_mutable_scalar || is_scalar_to_tensor;
}

bool AnfAlgo::IsDynamicSequence(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // Check if the node is dynamic sequence by sign in abstract.
  const auto &is_dynamic_len_func = [&node]() {
    const auto &abstract = node->abstract();
    if (abstract == nullptr || (!abstract->isa<abstract::AbstractSequence>())) {
      return false;
    }

    const auto &sequence_abstract = abstract->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(sequence_abstract);
    return sequence_abstract->dynamic_len() || sequence_abstract->dynamic_len_element_abs() != nullptr;
  };

  // Check if the node is dynamic sequence by sign in node, in cnode it is an attr in primitive, in parameter, it is
  // an sign.
  if (node->isa<Parameter>()) {
    const auto &parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    if (parameter->dynamic_len()) {
      return true;
    }
    bool is_dynamic = is_dynamic_len_func();
    if (is_dynamic) {
      parameter->set_dynamic_len(true);
    }
    return is_dynamic;
  } else if (node->isa<CNode>()) {
    if (IsCallNode(node)) {
      return is_dynamic_len_func();
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->HasAttr(kAttrDynamicLenName)) {
      return GetValue<bool>(cnode->GetAttr(kAttrDynamicLenName));
    } else {
      bool is_dynamic = is_dynamic_len_func();
      cnode->AddAttr(kAttrDynamicLenName, MakeValue(is_dynamic));
      return is_dynamic;
    }
  } else if (node->isa<ValueNode>()) {
    return is_dynamic_len_func();
  }
  return false;
}

bool AnfAlgo::IsAnyTypeOutput(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->isa<CNode>()) {
    if (IsCallNode(node)) {
      if (node->abstract() != nullptr && node->abstract()->isa<abstract::AbstractAny>()) {
        return true;
      }
      return false;
    }
    const auto &cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    if (cnode->HasAttr(kAttrAnyOutputName)) {
      return GetValue<bool>(cnode->GetAttr(kAttrAnyOutputName));
    } else {
      bool is_any_output = (node->abstract() != nullptr && node->abstract()->isa<abstract::AbstractAny>());
      cnode->AddAttr(kAttrAnyOutputName, MakeValue(is_any_output));
      return is_any_output;
    }
  }
  return false;
}

namespace {
bool IsIncludeAny(const abstract::AbstractBasePtr &abstract) {
  if (abstract == nullptr) {
    return false;
  }
  if (abstract->isa<abstract::AbstractAny>()) {
    return true;
  }
  if (!abstract->isa<abstract::AbstractSequence>()) {
    return false;
  }
  const auto &seq_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abstract);
  if (std::any_of(seq_abstract->elements().begin(), seq_abstract->elements().end(),
                  [](const auto &abstract) { return IsIncludeAny(abstract); })) {
    return true;
  }
  return false;
}
}  // namespace

bool AnfAlgo::IsAnyTypeInput(const std::vector<AnfNodePtr> &inputs) {
  for (const auto &input : inputs) {
    MS_EXCEPTION_IF_NULL(input);
    if (IsIncludeAny(input->abstract())) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::HasTupleInput(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t input_num = node->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(node, i);
    MS_EXCEPTION_IF_NULL(input_node);
    if (common::AnfAlgo::IsTupleOutput(input_node)) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::HasDynamicTupleInput(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  size_t input_num = node->size() - 1;
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node = common::AnfAlgo::GetInputNode(node, i);
    MS_EXCEPTION_IF_NULL(input_node);
    if (common::AnfAlgo::IsDynamicSequence(input_node)) {
      return true;
    }
  }
  return false;
}

bool AnfAlgo::IsReduceOp(const std::string &op_name) {
  static const std::set<std::string> reduce_op_type = {prim::kPrimReduceAll->name(),  prim::kPrimReduceAny->name(),
                                                       prim::kPrimReduceMean->name(), prim::kPrimReduceMax->name(),
                                                       prim::kPrimReduceMin->name(),  prim::kPrimReduceProd->name(),
                                                       prim::kPrimReduceSum->name(),  prim::kPrimSquareSumV1->name()};
  return reduce_op_type.find(op_name) != reduce_op_type.end();
}

bool AnfAlgo::IsTypeTransformOp(const std::string &op_name) {
  static const std::set<std::string> type_trans_op_names = {
    prim::kPrimTupleToTensor->name(),  prim::kPrimTensorToTuple->name(), prim::kPrimScalarToTensor->name(),
    prim::kPrimTensorToScalar->name(), prim::kPrimRealMakeTuple->name(), prim::kPrimRealTupleGetItem->name()};
  return type_trans_op_names.find(op_name) != type_trans_op_names.end();
}

abstract::BaseShapePtr AnfAlgo::GetDynamicSequenceShape(const AnfNodePtr &node, size_t output_idx) {
  MS_EXCEPTION_IF_NULL(node);
  abstract::AbstractSequencePtr sequence_abs = nullptr;
  if (node->Shape() == nullptr || (!node->Shape()->isa<abstract::DynamicSequenceShape>())) {
    MS_LOG(INFO) << "node:" << node->fullname_with_scope() << " index:" << output_idx
                 << " abs:" << node->abstract()->ToString();
    if (!node->abstract()->isa<abstract::AbstractSequence>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Not sequence abstract in node:" << node->DebugString()
                                 << " for dynamic sequence shape.";
    }
    const auto &top_sequence_abs = node->abstract()->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(top_sequence_abs);
    if (output_idx >= top_sequence_abs->elements().size()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid index:" << output_idx << " for abs:" << top_sequence_abs->ToString()
                                 << "node:" << node->fullname_with_scope();
    }
    const auto &sub_abs = top_sequence_abs->elements()[output_idx];
    MS_EXCEPTION_IF_NULL(sub_abs);
    if (!sub_abs->isa<abstract::AbstractSequence>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Not sequence abstract in node:" << node->DebugString()
                                 << " for dynamic sequence shape.";
    }
    sequence_abs = sub_abs->cast<abstract::AbstractSequencePtr>();
  } else {
    if (node->abstract() == nullptr) {
      MS_LOG(INTERNAL_EXCEPTION) << "Empty abstract in node:" << node->DebugString() << " for dynamic sequence shape.";
    }
    if (!node->abstract()->isa<abstract::AbstractSequence>()) {
      MS_LOG(INTERNAL_EXCEPTION) << "Not sequence abstract in node:" << node->DebugString()
                                 << " for dynamic sequence shape.";
    }
    sequence_abs = node->abstract()->cast<abstract::AbstractSequencePtr>();
  }
  MS_EXCEPTION_IF_NULL(sequence_abs);
  if (!sequence_abs->dynamic_len()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Not dynamic abstract in node:" << node->DebugString()
                               << " for dynamic sequence shape.";
  }
  const auto &element_abs = sequence_abs->dynamic_len_element_abs();
  if (element_abs == nullptr) {
    MS_LOG(INFO) << "No element abs for node:" << node->DebugString() << " index:" << output_idx;
    ShapeVector empty_shape{0};
    return std::make_shared<abstract::Shape>(empty_shape);
  }
  return element_abs->BuildShape();
}

abstract::AbstractBasePtr AnfAlgo::FetchAbstractByIndex(const AbstractBasePtr &abstract, size_t index) {
  MS_EXCEPTION_IF_NULL(abstract);
  if (!abstract->isa<abstract::AbstractSequence>() || abstract->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
    if (index != 0) {
      MS_LOG(INTERNAL_EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
    }
    return abstract;
  }

  auto tuple_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(tuple_abstract);
  const auto &sub_abstracts = tuple_abstract->elements();
  size_t real_index = index;
  for (const auto &sub_abstract : sub_abstracts) {
    size_t tmp_index = common::AnfAlgo::GetOutputNumByAbstract(sub_abstract);
    if (real_index >= tmp_index) {
      real_index -= tmp_index;
      continue;
    }
    return FetchAbstractByIndex(sub_abstract, real_index);
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Invalid abstract index:" << index << " for abstract:" << abstract->ToString();
}

std::string AnfAlgo::GetInputName(const CNodePtr &origin_op, size_t input_index) {
  auto prim_func_input_name = ops::GetInputNameByIndex(GetCNodeName(origin_op), input_index);
  if (prim_func_input_name != "") {
    return prim_func_input_name;
  }
  auto origin_primitive = GetCNodePrimitive(origin_op);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  auto input_names = origin_primitive->GetAttr(kAttrInputNames);
  if (input_names == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "input_names are nullptr in cnode " << origin_op->fullname_with_scope()
                               << ", debug string:" << origin_op->DebugString()
                               << ", attr text:" << origin_primitive->GetAttrsText();
  }

  auto input_names_vec = GetValue<std::vector<std::string>>(input_names);
  if (input_index >= input_names_vec.size()) {
    MS_LOG(INFO) << "Input index is invalid. input index: " << input_index << ", input name size "
                 << input_names_vec.size();
    return "";
  }
  return input_names_vec[input_index];
}

bool AnfAlgo::IsNoOuputNode(const AnfNodePtr &node) {
  const mindspore::HashSet<PrimitivePtr, PrimitiveHasher, PrimitiveEqual> no_output_prims = {
    prim::kPrimSend,
    prim::kPrimNPUClearFloatStatusV2,
    prim::kPrimInitPartitionMap,
    prim::kPrimInitEmbeddingHashmap,
    prim::kPrimEmbeddingTableImport,
    prim::kPrimEmbeddingComputeVarExport,
    prim::kPrimEmbeddingComputeVarImport,
    prim::kPrimEmbeddingTableExport};
  if (IsOneOfPrimitiveCNode(node, no_output_prims)) {
    return true;
  }
  return false;
}

ValuePtr AnfAlgo::ValueToScalar(const ValuePtr &value, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<KernelTensorValue>()) {
    return nullptr;
  }
  const auto &kernel_tensor_value = value->cast<KernelTensorValuePtr>();
  MS_EXCEPTION_IF_NULL(kernel_tensor_value);
  MS_EXCEPTION_IF_NULL(kernel_tensor_value->GetDataPtr());
  switch (type_id) {
    case kNumberTypeBool:
      return MakeValue(*reinterpret_cast<const bool *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeInt16:
      return MakeValue(*reinterpret_cast<const int16_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeUInt16:
      return MakeValue(*reinterpret_cast<const uint16_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeInt8:
      return MakeValue(*reinterpret_cast<const int8_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeUInt8:
      return MakeValue(*reinterpret_cast<const uint8_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeInt32:
      return MakeValue(*reinterpret_cast<const int32_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeUInt32:
      return MakeValue(*reinterpret_cast<const uint32_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeInt64:
      return MakeValue(*reinterpret_cast<const int64_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeUInt64:
      return MakeValue(*reinterpret_cast<const uint64_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeFloat16:
      return MakeValue(*reinterpret_cast<const uint16_t *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeFloat32:
      return MakeValue(*reinterpret_cast<const float *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeFloat64:
      return MakeValue(*reinterpret_cast<const double *>(kernel_tensor_value->GetDataPtr()));
    case kNumberTypeBFloat16:
      return MakeValue(*reinterpret_cast<const uint16_t *>(kernel_tensor_value->GetDataPtr()));
    default:
      MS_LOG(DEBUG) << "Not support scalar type:" << type_id;
  }
  return nullptr;
}

namespace {
void IterateFindTensor(ValuePtrList *value_list, const VectorRef &ref_list) {
  MS_EXCEPTION_IF_NULL(value_list);
  for (size_t i = 0; i < ref_list.size(); ++i) {
    if (utils::isa<tensor::BaseTensorPtr>(ref_list[i])) {
      auto tensor_ptr = utils::cast<std::shared_ptr<tensor::BaseTensor>>(ref_list[i]);
      MS_EXCEPTION_IF_NULL(tensor_ptr);
      (void)value_list->emplace_back(tensor_ptr);
    } else if (utils::isa<VectorRef>(ref_list[i])) {
      auto ref_iter = utils::cast<VectorRef>(ref_list[i]);
      IterateFindTensor(value_list, ref_iter);
    } else if (utils::isa<tensor::CSRTensorPtr>(ref_list[i])) {
      auto csr_tensor = utils::cast<tensor::CSRTensorPtr>(ref_list[i]);
      MS_EXCEPTION_IF_NULL(csr_tensor);
      (void)value_list->emplace_back(csr_tensor);
    } else {
      MS_LOG(EXCEPTION) << "The ref value " << ref_list[i].ToString() << " is not a vector ref or a tensor!";
    }
  }
}

bool HasAbstractFunction(const AbstractBasePtr &abs) {
  if (abs->isa<abstract::AbstractSequence>() && !abs->isa<abstract::AbstractSparseTensor>()) {
    auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq->dynamic_len()) {
      return HasAbstractFunction(abs_seq->dynamic_len_element_abs());
    }
    return std::any_of(abs_seq->elements().cbegin(), abs_seq->elements().cend(), HasAbstractFunction);
  }
  // if abs it not AbstractSequence.
  return abs->isa<abstract::AbstractFunction>();
}

bool IsCellReuse(const AnfNodePtr &input) {
  if (IsValueNode<FuncGraph>(input)) {
    auto fg = GetValueNode<FuncGraphPtr>(input);
    MS_EXCEPTION_IF_NULL(fg);
    if (fg->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
      return true;
    }
  }
  return false;
}

bool AcceptableReturnValue(const CNodePtr &cnode, const AnfNodePtr &input0) {
  if (IsCellReuse(input0)) {
    return true;
  }
  auto func_graphs = abstract::GetFuncGraphsFromCallNode(cnode);
  auto graph_has_function_output = [](const FuncGraphPtr &fg) { return HasAbstractFunction(fg->output()->abstract()); };
  if (std::all_of(func_graphs.cbegin(), func_graphs.cend(), std::not_fn(graph_has_function_output))) {
    return true;
  }
  return false;
}

bool SupportInlinePartial(const AnfNodePtr &input0) {
  // inline partial
  if (IsPrimitiveCNode(input0, prim::kPrimTupleGetItem)) {
    auto tuple_get_node = input0->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_get_node);
    auto get_from_node = tuple_get_node->input(1);
    auto idx = common::AnfAlgo::GetTupleGetItemOutIndex(tuple_get_node);
    MS_EXCEPTION_IF_NULL(get_from_node);
    // tuple get item from a call subgraph output
    if (get_from_node->isa<CNode>() && IsValueNode<FuncGraph>(get_from_node->cast<CNodePtr>()->input(0))) {
      auto call_graph = GetValueNode<FuncGraphPtr>(get_from_node->cast<CNodePtr>()->input(0));
      MS_EXCEPTION_IF_NULL(call_graph);
      auto graph_out = call_graph->output();
      MS_EXCEPTION_IF_NULL(graph_out);
      size_t tuple_input_num = common::AnfAlgo::GetInputTensorNum(graph_out);
      // the partial must be the last output
      if (graph_out->isa<CNode>() && tuple_input_num == idx + 1) {
        int partial_cnt = 0;
        for (size_t i = 0; i < tuple_input_num; i++) {
          auto input = graph_out->cast<CNodePtr>()->input(i + 1);
          if (IsPrimitiveCNode(input, prim::kPrimPartial)) {
            partial_cnt++;
          }
        }
        auto partial = graph_out->cast<CNodePtr>()->input(idx + 1);
        MS_EXCEPTION_IF_NULL(partial);
        // we only support one partial func at the last return value now
        if (partial_cnt != 1 || !IsPrimitiveCNode(partial, prim::kPrimPartial)) {
          if (partial_cnt != 0) {
            MS_LOG(INFO) << "Partial func cnt: " << partial_cnt
                         << ", last return value: " << partial->fullname_with_scope();
          }
          return false;
        }
        auto partial_inputs = partial->cast<CNodePtr>()->inputs();
        // the input of partial can't be FuncGraph/Partial
        bool has_illegal_input = std::any_of(
          partial_inputs.begin() + kPartialMinInputSize, partial_inputs.end(), [](const AnfNodePtr &partial_input) {
            return IsValueNode<FuncGraph>(partial_input) || IsPrimitiveCNode(partial_input, prim::kPrimPartial);
          });
        return !has_illegal_input;
      }
    }
  }
  return false;
}
}  // namespace

ValuePtrList AnfAlgo::TransformVectorRefToMultiValue(const VectorRef &base_ref) {
  ValuePtrList value_list;
  if (utils::isa<VectorRef>(base_ref)) {
    auto ref_list = utils::cast<VectorRef>(base_ref);
    IterateFindTensor(&value_list, ref_list);
  } else if (utils::isa<tensor::Tensor>(base_ref)) {
    auto tensor_ptr = utils::cast<std::shared_ptr<tensor::Tensor>>(base_ref);
    MS_EXCEPTION_IF_NULL(tensor_ptr);
    (void)value_list.emplace_back(tensor_ptr);
  } else {
    MS_LOG(EXCEPTION) << "The ref value " << base_ref.ToString() << " is not a vector ref or a tensor!";
  }
  return value_list;
}

bool AnfAlgo::HasIncorporateCallNode(const CNodePtr &cnode) {
  if (!IsValueNode<Primitive>(cnode->input(0))) {  // If cnode is a call node.
    auto input0 = cnode->input(0);
    if (IsPrimitiveCNode(input0, prim::kPrimSwitch) || IsPrimitiveCNode(input0, prim::kPrimSwitchLayer) ||
        IsValueNode<FuncGraph>(input0)) {
      if (IsCellReuse(input0) && IsEnableRefMode()) {
        MS_LOG(INFO) << "Use cell reuse when enable ge mode: " << cnode->DebugString();
        return true;
      }
      if (AcceptableReturnValue(cnode, input0)) {
        return false;
      }
    }
    if (SupportInlinePartial(input0)) {
      return false;
    }
    MS_LOG(INFO) << "Call has indirect call: " << cnode->DebugString();
    return true;
  }
  return false;
}

bool AnfAlgo::IsDynamicGraph(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::vector<AnfNodePtr> node_list = TopoSort(func_graph->get_return(), SuccDeeperSimple);
  AnfNodePtr dynamic_node = nullptr;
  AnfNodePtr pyexecute_node = nullptr;
  for (const auto &node : node_list) {
    if (node->abstract() == nullptr) {
      MS_LOG(INFO) << "Null abstract of node: " << node->DebugString();
      continue;
    }
    if (node->abstract() != nullptr) {
      auto shape = node->abstract()->GetShape();
      // Dynamic shape tensor.
      if (shape->isa<abstract::TensorShape>() && IsDynamic(shape->GetShapeVector())) {
        dynamic_node = node;
        break;
      }
      // Dynamic len sequence.
      if (node->abstract()->isa<abstract::AbstractSequence>() &&
          node->abstract()->cast<abstract::AbstractSequencePtr>()->dynamic_len()) {
        dynamic_node = node;
        break;
      }
      // PyExecute node exist
      if (IsPrimitiveCNode(node, prim::kPrimPyExecute)) {
        pyexecute_node = node;
      }
    }
  }
  if (dynamic_node != nullptr) {
    MS_LOG(INFO) << "Func graph:" << func_graph->ToString()
                 << " is dynamic shape graph, because find dynamic shape node:" << dynamic_node->DebugString()
                 << ", abstract: " << dynamic_node->abstract()->ToString();
    return true;
  }
  if (pyexecute_node != nullptr) {
    MS_LOG(INFO) << "Func graph:" << func_graph->ToString() << " has pyexecute node:" << pyexecute_node->DebugString();
    return true;
  }
  return false;
}
}  // namespace common
}  // namespace mindspore
