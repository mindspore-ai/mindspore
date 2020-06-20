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
#include "pre_activate/ascend/enhancer/insert_memcpy_async_for_hccl_op.h"
#include <vector>
#include <set>
#include <string>
#include "utils/utils.h"
#include "session/anf_runtime_algorithm.h"
#include "optimizer/opt.h"
#include "pre_activate/ascend/ascend_helper.h"

namespace mindspore {
namespace opt {
namespace {
// insert memcpy for some cnode even if not a Ref cnode
const std::set<std::string> kNeedInsertMemcpyOpSet = {kLambNextMVOpName, kLambNextMVWithDecayOpName,
                                                      kLambUpdateWithLROpName};

bool IsParameterOrValueNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  auto kernel_with_index = AnfAlgo::VisitKernelWithReturnType(node, 0, true);
  return kernel_with_index.first->isa<Parameter>() || kernel_with_index.first->isa<ValueNode>();
}

void TransferControl(const CNodePtr &hccl_node, const AnfNodePtr &memcpy_async, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(hccl_node);
  MS_EXCEPTION_IF_NULL(memcpy_async);
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
    int output_index = node_index.second;
    if (AnfAlgo::CheckPrimitiveType(output, prim::kPrimControlDepend)) {
      CNodePtr control_depend = output->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(control_depend);
      std::vector<AnfNodePtr> new_inputs;
      for (size_t i = 0; i < control_depend->size(); ++i) {
        if (i == IntToSize(output_index)) {
          new_inputs.push_back(memcpy_async);
        } else {
          new_inputs.push_back(control_depend->input(i));
        }
      }
      control_depend->set_inputs(new_inputs);
    }
  }
}
}  // namespace

bool InsertMemcpyAsyncForHcclOp::NeedInsertMemcpy(const FuncGraphPtr &graph, const AnfNodePtr &input) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(input);
  // when input is a parameter or is a value node
  if (IsParameterOrValueNode(input)) {
    return true;
  }

  // when input is a Ref or some special cnodes
  if (kernel_query_->IsTbeRef(input) ||
      kNeedInsertMemcpyOpSet.find(AnfAlgo::GetCNodeName(input)) != kNeedInsertMemcpyOpSet.end()) {
    return true;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users = manager->node_users();
  auto iter = node_users.find(input);
  if (iter == node_users.end()) {
    MS_LOG(EXCEPTION) << "node has no output in manager";
  }
  // when input is used by others
  if (iter->second.size() > 1) {
    return true;
  }
  return false;
}

void InsertMemcpyAsyncForHcclOp::InsertMemcpyAsync(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  bool has_insert_memcpy = false;
  AnfNodePtr memcpy_async = nullptr;
  std::vector<AnfNodePtr> new_inputs = {hccl_node->input(0)};
  for (size_t i = 1; i < hccl_node->size(); ++i) {
    auto input = hccl_node->input(i);
    if (NeedInsertMemcpy(graph, input)) {
      memcpy_async = CreateMemcpyAsyncOp(graph, input);
      has_insert_memcpy = true;
      new_inputs.push_back(memcpy_async);
    } else {
      new_inputs.push_back(input);
    }
  }

  if (has_insert_memcpy) {
    CNodePtr new_hccl_node = std::make_shared<CNode>(*hccl_node);
    new_hccl_node->set_inputs(new_inputs);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    MS_LOG(DEBUG) << "start replace new_hccl_node to old hccl_node";
    (void)manager->Replace(hccl_node, new_hccl_node);
    MS_LOG(DEBUG) << "end replace";

    // transer hccl op's control to the memcpy_async
    if (hccl_node->size() == 2) {
      TransferControl(new_hccl_node, memcpy_async, graph);
    }
  }
}

const AnfNodePtr InsertMemcpyAsyncForHcclOp::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  if (func_graph == nullptr || node == nullptr || !node->isa<CNode>()) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  if (!AnfAlgo::IsCommunicationOp(node)) {
    return nullptr;
  }
  InsertMemcpyAsync(func_graph, cnode);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
