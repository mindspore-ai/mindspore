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
#include "backend/optimizer/pass/sparse_process.h"
#include "ir/anf.h"
#include "utils/convert_utils.h"
#include "utils/anf_utils.h"
#include "backend/optimizer/common/helper.h"
#include "backend/optimizer/common/optimizer.h"
#include "backend/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace opt {
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

const AnfNodePtr SparseProcess::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                        const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  std::string prim_name = prim->name();
  auto kernel_graph = func_graph->cast<std::shared_ptr<session::KernelGraph>>();
  // cnode is a MakeSparse node
  if (make_sparse_set.find(prim_name) != make_sparse_set.end()) {
    std::vector<AnfNodePtr> inputs;
    inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    // Inputs of node should be [make_sparse, indices, values, dense_shape], so offset by 1 to get items;
    (void)inputs.insert(inputs.end(), cnode->inputs().begin() + 1, cnode->inputs().end());
    auto new_node = cnode->func_graph()->NewCNode(inputs);
    auto abs_sparse = dyn_cast<abstract::AbstractCSRTensor>(node->abstract());
    std::vector<AbstractBasePtr> abstract_list{abs_sparse->indptr(), abs_sparse->indices(), abs_sparse->values(),
                                               abs_sparse->dense_shape()};
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
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
