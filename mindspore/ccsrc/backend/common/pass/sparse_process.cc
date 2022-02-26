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
    return true;
  }
  return false;
}

bool SplitCNode(const AnfNodePtr &node, std::vector<AnfNodePtr> *new_inputs) {
  auto cnode = node->cast<CNodePtr>();
  auto sparse_prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(sparse_prim);
  // Currently, only MakeCSR and MakeTuple nodes can be split.
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

std::vector<AbstractBasePtr> GetAbstractList(const AnfNodePtr &node, const std::string &prim_name) {
  MS_EXCEPTION_IF_NULL(node);
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
  // cnode is a MakeSparse node
  if (make_sparse_set.find(prim_name) != make_sparse_set.end()) {
    std::vector<AnfNodePtr> inputs;
    inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    (void)inputs.insert(inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
    auto new_node = cnode->func_graph()->NewCNode(inputs);
    std::vector<AbstractBasePtr> abstract_list = GetAbstractList(node, prim_name);
    auto abs_res = std::make_shared<abstract::AbstractTuple>(abstract_list);
    new_node->set_abstract(abs_res);
    new_node->set_scope(cnode->scope());
    if (kernel_graph != nullptr) {
      kernel_graph->FrontBackendlMapUpdate(cnode, new_node);
    }
    return new_node;
    // cnode is a SparseGetAttr node
  } else if (sparse_attr_map.find(prim_name) != sparse_attr_map.end()) {
    const auto &inputs = cnode->inputs();
    // Inputs should be [sparse_getattr, sparse]
    if (inputs.size() <= 1) {
      MS_LOG_EXCEPTION << "For SparseGetAttr, CNode must have 2 inputs (Prim, Sparse)";
    }
    constexpr size_t sparse_index = 1;
    AbstractCSRToAbstractTuple(inputs[sparse_index]);
    int64_t index = sparse_attr_map.at(prim_name);
    auto cons_node = NewValueNode(index);
    AbstractBasePtr aptr = std::make_shared<abstract::AbstractScalar>(std::make_shared<Int64Imm>(index));
    cons_node->set_abstract(aptr);
    auto new_node = NewCNode({NewValueNode(prim::kPrimTupleGetItem), inputs[sparse_index], cons_node}, func_graph);
    new_node->set_abstract(node->abstract());
    return new_node;
    // ComputeSparse node: SparseTensorDenseMatmul, CSRDenseMul, CSRReduceSum
  } else if (sparse_op_set.find(prim_name) != sparse_op_set.end()) {
    const auto &inputs = cnode->inputs();
    std::vector<AnfNodePtr> new_inputs;
    new_inputs.push_back(inputs[0]);
    for (size_t i = 1; i < inputs.size(); ++i) {
      if (inputs[i]->isa<CNode>()) {
        if (SplitCNode(inputs[i], &new_inputs)) continue;
      } else if (inputs[i]->isa<ValueNode>()) {
        if (SplitValueNode(inputs[i], &new_inputs, kernel_graph)) continue;
      } else if (inputs[i]->isa<Parameter>()) {
        if (SplitParameter(inputs[i], &new_inputs, kernel_graph)) continue;
      }
      new_inputs.push_back(inputs[i]);
    }
    auto new_node = cnode->func_graph()->NewCNode(new_inputs);
    new_node->set_abstract(node->abstract());
    return new_node;
  }

  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
