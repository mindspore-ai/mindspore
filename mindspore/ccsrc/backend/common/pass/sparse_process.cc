/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <vector>
#include <memory>
#include "backend/common/pass/sparse_process.h"
#include "ir/anf.h"
#include "include/common/utils/convert_utils.h"
#include "utils/anf_utils.h"
#include "backend/common/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace opt {
using CSRTensor = mindspore::tensor::CSRTensor;
using CSRTensorPtr = mindspore::tensor::CSRTensorPtr;

constexpr auto kCSRValueNodeNum = 2;
constexpr auto kSparseAttrIndex = 1;

// Convert CSRTensor Parameter or ValueNode to Tuple by setting its abstract.
void AbstractCSRToAbstractTuple(const AnfNodePtr &sparse) {
  MS_EXCEPTION_IF_NULL(sparse);
  if (!(sparse->isa<Parameter>() || sparse->isa<ValueNode>())) {
    return;
  }
  auto param_abs = sparse->abstract();
  MS_EXCEPTION_IF_NULL(param_abs);
  if (param_abs->isa<abstract::AbstractCSRTensor>()) {
    auto abs_sparse = param_abs->cast<abstract::AbstractCSRTensorPtr>();
    std::vector<AbstractBasePtr> abstract_list{abs_sparse->indptr(), abs_sparse->indices(), abs_sparse->values(),
                                               abs_sparse->dense_shape()};
    auto abs_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
    abs_tuple->set_type(abs_tuple->BuildType());
    sparse->set_abstract(abs_tuple);
  }
}

ValueNodePtr MakeNewValueNodeToGraph(const ValueNodePtr &val, const AbstractBasePtr &abs,
                                     const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto node = kernel_graph->NewValueNode(val);
  MS_EXCEPTION_IF_NULL(node);
  node->set_abstract(abs);
  kernel_graph->AddValueNodeToGraph(node);
  return node;
}

bool SplitValueNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *new_inputs, const KernelGraphPtr &kernel_graph) {
  ValuePtr value = node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(value);
  if (!value->isa<CSRTensor>()) return false;
  auto csr_tensor = value->cast<CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto csr_abs = node->abstract()->cast<abstract::AbstractCSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_abs);
  auto new_indptr = MakeNewValueNodeToGraph(NewValueNode(csr_tensor->GetIndptr()), csr_abs->indptr(), kernel_graph);
  new_inputs->push_back(new_indptr);
  auto new_indices = MakeNewValueNodeToGraph(NewValueNode(csr_tensor->GetIndices()), csr_abs->indices(), kernel_graph);
  new_inputs->push_back(new_indices);
  auto new_values = MakeNewValueNodeToGraph(NewValueNode(csr_tensor->GetValues()), csr_abs->values(), kernel_graph);
  new_inputs->push_back(new_values);
  return true;
}

bool SplitParameter(const AnfNodePtr &node, std::vector<AnfNodePtr> *new_inputs, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(node);
  auto node_abs = node->abstract();
  MS_EXCEPTION_IF_NULL(node_abs);
  static HashMap<AnfNodePtr, std::vector<AnfNodePtr>> csr_params_map;
  auto param = node->cast<ParameterPtr>();
  MS_EXCEPTION_IF_NULL(param);
  if (node_abs->isa<abstract::AbstractCSRTensor>()) {
    auto param_abs = node_abs->cast<abstract::AbstractCSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(param_abs);
    MS_EXCEPTION_IF_NULL(param_abs->indptr());
    MS_EXCEPTION_IF_NULL(param_abs->indices());
    MS_EXCEPTION_IF_NULL(param_abs->values());
    auto new_indptr =
      MakeNewValueNodeToGraph(NewValueNode(param_abs->indptr()->BuildValue()), param_abs->indptr(), kernel_graph);
    MS_EXCEPTION_IF_NULL(new_indptr);
    new_inputs->push_back(new_indptr);
    auto new_indices =
      MakeNewValueNodeToGraph(NewValueNode(param_abs->indices()->BuildValue()), param_abs->indices(), kernel_graph);
    MS_EXCEPTION_IF_NULL(new_indices);
    new_inputs->push_back(new_indices);
    // Set CSRTensor Parameter abstract to Tensor by its values.
    node->set_abstract(param_abs->values()->Broaden());
    new_inputs->push_back(node);
    // Set csr_params_map
    if (csr_params_map.find(node) == csr_params_map.end()) {
      csr_params_map[node].emplace_back(new_indptr);
      csr_params_map[node].emplace_back(new_indices);
    }
    return true;
    // If the cnode has a csr_tensor_param which has been split, use the map to find its indptr and indices.
  } else if (node_abs->isa<abstract::AbstractTensor>() && csr_params_map.find(node) != csr_params_map.end()) {
    if (csr_params_map[node].size() != kCSRValueNodeNum) {
      MS_LOG(ERROR) << "csr_params_map[" << node->DebugString() << "] has " << csr_params_map[node].size()
                    << " inputs, but expect two inputs! They are all added in new_inputs.";
    }
    new_inputs->insert(new_inputs->end(), csr_params_map[node].begin(), csr_params_map[node].end());
    new_inputs->push_back(node);
    return true;
  }
  return false;
}

bool SplitCNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *new_inputs) {
  auto cnode = node->cast<CNodePtr>();
  auto sparse_prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(sparse_prim);
  // Currently, only MakeCSR/MakeCOO and MakeTuple nodes can be split.
  if (make_sparse_set.count(sparse_prim->name()) <= 0 && sparse_prim->name().compare(prim::kPrimMakeTuple->name()) != 0)
    return false;

  auto sparse_inputs = cnode->inputs();
  // skip the last input, as it always represents shape, and has already been
  // registered as primitive attribute.
  for (size_t j = 1; j < sparse_inputs.size() - 1; ++j) {
    new_inputs->push_back(sparse_inputs[j]);
  }
  return true;
}

std::vector<AbstractBasePtr> GetAbstractList(const AnfNodePtr &node, const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(prim);
  std::string prim_name = prim->name();
  if (prim_name == prim::kPrimMakeCSRTensor->name()) {
    auto abs_sparse = dyn_cast<abstract::AbstractCSRTensor>(node->abstract());
    MS_EXCEPTION_IF_NULL(abs_sparse);
    return {abs_sparse->indptr(), abs_sparse->indices(), abs_sparse->values(), abs_sparse->dense_shape()};
  } else if (prim_name == prim::kPrimMakeCOOTensor->name()) {
    auto abs_sparse = dyn_cast<abstract::AbstractCOOTensor>(node->abstract());
    MS_EXCEPTION_IF_NULL(abs_sparse);
    return {abs_sparse->indices(), abs_sparse->values(), abs_sparse->dense_shape()};
  }
  return {};
}

CNodePtr ConvertMakeSparseToMakeTuple(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  std::vector<AnfNodePtr> inputs;
  inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
  (void)inputs.insert(inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());

  auto new_node = NewCNode(inputs, cnode->func_graph());
  std::vector<AbstractBasePtr> abstract_list = GetAbstractList(node, common::AnfAlgo::GetCNodePrimitive(cnode));
  auto abs_res = std::make_shared<abstract::AbstractTuple>(abstract_list);
  new_node->set_abstract(abs_res);
  new_node->set_scope(cnode->scope());
  if (kernel_graph != nullptr) {
    kernel_graph->FrontBackendlMapUpdate(cnode, new_node);
  }
  return new_node;
}

CNodePtr ConvertSparseGetAttrToTupleGetItem(int64_t index, const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  const auto &inputs = cnode->inputs();
  if (inputs.size() <= kSparseAttrIndex) {
    MS_LOG(EXCEPTION) << "For SparseGetAttr, CNode must have 2 inputs (Prim, Sparse)";
  }
  AbstractCSRToAbstractTuple(inputs[kSparseAttrIndex]);
  auto index_node = NewValueNode(index);
  AbstractBasePtr index_abs = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(index));
  index_node->set_abstract(index_abs);
  auto new_node =
    NewCNode({NewValueNode(prim::kPrimTupleGetItem), inputs[kSparseAttrIndex], index_node}, cnode->func_graph());
  new_node->set_abstract(node->abstract());
  if (kernel_graph != nullptr) {
    kernel_graph->FrontBackendlMapUpdate(cnode, new_node);
  }
  return new_node;
}

CNodePtr FetchInputsForSparseOP(const AnfNodePtr &node, const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  CNodePtr cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);

  if (cnode->GetAttr("has_been_split") != nullptr) {
    MS_LOG(INFO) << "Do not process CNode " << cnode << " (" << cnode->DebugString() << "), because it has been split.";
    return nullptr;
  }
  const auto &inputs = cnode->inputs();
  std::vector<AnfNodePtr> new_inputs;
  new_inputs.push_back(inputs[0]);
  for (size_t i = 1; i < inputs.size(); ++i) {
    if (inputs[i]->isa<CNode>()) {
      if (SplitCNode(inputs[i], &new_inputs)) continue;
    } else if (inputs[i]->isa<ValueNode>()) {
      if (SplitValueNode(inputs[i], &new_inputs, kernel_graph)) continue;
    } else if (inputs[i]->isa<Parameter>()) {
      // 1. Split CSRTensor param to multiple tensors.
      // 2. Set CSRTensor abstract to AbstractTensor that is related its values.
      if (SplitParameter(inputs[i], &new_inputs, kernel_graph)) continue;
    }
    new_inputs.push_back(inputs[i]);
  }
  auto new_node = NewCNode(new_inputs, cnode->func_graph());
  new_node->set_abstract(node->abstract());
  // Set attr "has_been_split" to prevent the node is split more than once.
  new_node->AddAttr("has_been_split", MakeValue(true));
  if (kernel_graph != nullptr) {
    kernel_graph->FrontBackendlMapUpdate(cnode, new_node);
  }
  return new_node;
}

const AnfNodePtr SparseProcess::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  std::string prim_name = prim->name();
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  if (make_sparse_set.find(prim_name) != make_sparse_set.end()) {
    return ConvertMakeSparseToMakeTuple(node, kernel_graph);
  } else if (sparse_attr_map.find(prim_name) != sparse_attr_map.end()) {
    return ConvertSparseGetAttrToTupleGetItem(sparse_attr_map.at(prim_name), node, kernel_graph);
  } else if (sparse_op_set.find(prim_name) != sparse_op_set.end()) {
    return FetchInputsForSparseOP(node, kernel_graph);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
