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
#include "utils/trace_base.h"
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

// NodeUsersMap, for node B input i use node A, it will be one item in map with key: A, and value: (B, i)
bool IsNodeOutPutUsedByOtherRealKernel(const AnfNodeIndexSet &node_users) {
  if (node_users.size() == 1) {
    MS_LOG(INFO) << "This node only used once, no need to insert memcpy node.";
    return false;
  }
  for (const auto &node_pair : node_users) {
    auto node = node_pair.first;
    if (AnfAlgo::IsRealKernel(node) && !AnfAlgo::IsCommunicationOp(node)) {
      MS_LOG(INFO) << "This node only used other real kernel: " << node->fullname_with_scope();
      return true;
    }
  }
  MS_LOG(INFO) << "This node used by other node, but the node is not real kernel, no need to insert memcpy node.";
  return false;
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
      MS_LOG(EXCEPTION) << "node has no output in manager"
                        << " trace: " << trace::DumpSourceLines(input);
    }
    if (IsNodeOutPutUsedByOtherRealKernel(iter->second)) {
      return true;
    }
  }
  return false;
}

void InsertMemcpyAsyncForHcclOp::InsertMemcpyAsync(const FuncGraphPtr &graph, const CNodePtr &hccl_node) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(hccl_node);
  bool need_memcpy_async = false;
  std::vector<AnfNodePtr> new_inputs = {hccl_node->input(0)};
  for (size_t i = 1; i < hccl_node->size(); ++i) {
    auto input = hccl_node->input(i);
    if (NeedInsertMemcpy(graph, input, hccl_node)) {
      auto memcpy_async = CreateMemcpyAsyncOp(graph, input);
      if (memcpy_async == nullptr) {
        MS_LOG(EXCEPTION) << "Create memcpy_async op failed.";
      }
      if (input->isa<CNode>() && AnfAlgo::IsNodeDynamicShape(input)) {
        AnfAlgo::SetNodeAttr(kAttrIsDynamicShape, MakeValue(true), memcpy_async);
      }
      new_inputs.push_back(memcpy_async);
      need_memcpy_async = true;
    } else {
      new_inputs.push_back(input);
    }
  }

  if (need_memcpy_async) {
    CNodePtr new_hccl_node = std::make_shared<CNode>(*hccl_node);
    new_hccl_node->set_inputs(new_inputs);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    MS_LOG(DEBUG) << "start replace new_hccl_node to old hccl_node";
    (void)manager->Replace(hccl_node, new_hccl_node);
    MS_LOG(DEBUG) << "end replace";
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
