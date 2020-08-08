/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "backend/optimizer/ascend/enhancer/insert_memcpy_async_for_hccl_op.h"
#include <vector>
#include <set>
#include <string>
#include "utils/utils.h"
#include "backend/session/anf_runtime_algorithm.h"
#include "frontend/optimizer/opt.h"
#include "backend/optimizer/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
// insert memcpy for some cnode even if not a Ref cnode
const std::set<std::string> kNeedInsertMemcpyOpSet = {kLambNextMVOpName, kLambNextMVWithDecayOpName,
                                                      kLambUpdateWithLROpName};

bool IsParameterOrValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
  auto real_node = kernel_with_index.first;
  MS_EXCEPTION_IF_NULL(real_node);
  if (real_node->isa<Parameter>()) {
    return true;
  }
  return real_node->isa<ValueNode>();
}

void SetInput(const CNodePtr &control_depend, const int index, const FuncGraphPtr &graph, const CNodePtr &hccl_node,
              const std::vector<AnfNodePtr> &memcpy_async_list) {
  MS_EXCEPTION_IF_NULL(control_depend);
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  std::vector<AnfNodePtr> make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  make_tuple_inputs.insert(make_tuple_inputs.end(), memcpy_async_list.begin(), memcpy_async_list.end());
  make_tuple_inputs.emplace_back(hccl_node);
  auto make_tuple = graph->NewCNode(make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple);
  control_depend->set_input(IntToSize(index), make_tuple);
}

void DealControlForGetitem(const CNodePtr &tuple_getitem, const FuncGraphPtr &graph, const CNodePtr &hccl_node,
                           const std::vector<AnfNodePtr> &memcpy_async_list) {
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(tuple_getitem);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  for (const auto &node_index : iter->second) {
    AnfNodePtr output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimControlDepend)) {
      SetInput(output->cast<CNodePtr>(), node_index.second, graph, hccl_node, memcpy_async_list);
    }
  }
}

void TransferControl(const CNodePtr &hccl_node, const std::vector<AnfNodePtr> &memcpy_async_list,
                     const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(hccl_node);
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(hccl_node);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  // find hccl_node's output which is a control depend
  for (const auto &node_index : iter->second) {
    AnfNodePtr output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimControlDepend)) {
      SetInput(output->cast<CNodePtr>(), node_index.second, graph, hccl_node, memcpy_async_list);
    } else if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimTupleGetItem)) {
      DealControlForGetitem(output->cast<CNodePtr>(), graph, hccl_node, memcpy_async_list);
    }
  }
}
}  // namespace

bool InsertMemcpyAsyncForHcclOp::NeedInsertMemcpy(const FuncGraphPtr &graph, const AnfNodePtr &input,
                                                  const CNodePtr &cur_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(cur_node);
  // when input is a parameter or is a value node
  if (IsParameterOrValueNode(input)) {
    return true;
  }

  if (input->isa<CNode>()) {
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto &node_users = manager->node_users();

    // when input is a Ref cnode
    if (kernel_query_->IsTbeRef(input)) {
      return true;
    }

    // when input is some special cnodes
    if (kNeedInsertMemcpyOpSet.find(AnfAlgo::GetCNodeName(input)) != kNeedInsertMemcpyOpSet.end()) {
      return true;
    }

    // when input is used by others
    auto iter = node_users.find(input);
    if (iter == node_users.end()) {
      MS_LOG(EXCEPTION) << "node has no output in manager";
    }
    if (iter->second.size() > 1) {
      return true;
    }
  }
  return false;
}

void InsertMemcpyAsyncForHcclOp::InsertMemcpyAsync(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  std::vector<AnfNodePtr> memcpy_async_list;
  std::vector<AnfNodePtr> new_inputs = {hccl_node->input(0)};
  for (size_t i = 1; i < hccl_node->size(); ++i) {
    auto input = hccl_node->input(i);
    if (NeedInsertMemcpy(graph, input, hccl_node)) {
      auto memcpy_async = CreateMemcpyAsyncOp(graph, input);
      new_inputs.push_back(memcpy_async);
      memcpy_async_list.push_back(memcpy_async);
    } else {
      new_inputs.push_back(input);
    }
  }

  if (!memcpy_async_list.empty()) {
    CNodePtr new_hccl_node = std::make_shared<CNode>(*hccl_node);
    new_hccl_node->set_inputs(new_inputs);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    MS_LOG(DEBUG) << "start replace new_hccl_node to old hccl_node";
    (void)manager->Replace(hccl_node, new_hccl_node);
    MS_LOG(DEBUG) << "end replace";

    // transer hccl op's control to the memcpy_async
    TransferControl(new_hccl_node, memcpy_async_list, graph);
  }
}

const AnfNodePtr InsertMemcpyAsyncForHcclOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  if (!AnfAlgo::IsCommunicationOp(node)) {
    return nullptr;
  }
  InsertMemcpyAsync(func_graph, node->cast<CNodePtr>());
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
