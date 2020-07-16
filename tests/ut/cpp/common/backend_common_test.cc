/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#include "common/backend_common_test.h"

#include <vector>
#include <string>
#include <memory>

#include "utils/log_adapter.h"
#include "frontend/operator/ops.h"
#include "debug/anf_ir_dump.h"
#include "backend/session/ascend_session.h"
#include "pipeline/jit/resource.h"
#include "pipeline/jit/action.h"
#include "ir/anf.h"
#include "ir/manager.h"

namespace mindspore {
namespace {
std::vector<AnfNodePtr> GetCNodeList(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return());
  std::vector<AnfNodePtr> lst;
  for (auto &node : nodes) {
    MS_LOG(INFO) << "nodes: " << node->DebugString(10);
    if (node->isa<CNode>() && IsValueNode<Primitive>(node->cast<CNodePtr>()->input(0)) &&
        !IsPrimitiveCNode(node, prim::kPrimReturn)) {
      MS_LOG(INFO) << "push in anf_node list: " << node->DebugString(10);
      lst.push_back(node);
    }
  }
  return lst;
}
}  // namespace

bool BackendCommon::CheckEqualGraph(const FuncGraphPtr &a, const FuncGraphPtr &b) {
  FuncGraphPairMapEquiv equiv_graph_;
  NodeMapEquiv equiv_node_;
  return Isomorphic(a, b, &equiv_graph_, &equiv_node_);
}

std::shared_ptr<session::KernelGraph> BackendCommon::GetKernelGraph(const FuncGraphPtr &func_graph,
                                                                    const AbstractBasePtrList &args_spec_list,
                                                                    bool need_infer) {
  FuncGraphPtr inferred_graph = func_graph;
  if (need_infer) {
    inferred_graph = GetFuncGraph(func_graph, args_spec_list);
  }
  AnfNodePtrList applies = GetCNodeList(inferred_graph);
  AnfNodePtrList ins = inferred_graph->parameters();
  AnfNodePtrList outs = {inferred_graph->get_return()->input(1)};
  auto session = std::make_shared<session::AscendSession>();
  session->Init(0);
  auto kernel_graph = session->ConstructKernelGraph(applies, outs);
  kernel_graph->SetExecOrderByDefault();
  return kernel_graph;
}

FuncGraphPtr BackendCommon::GetFuncGraph(const FuncGraphPtr &func_graph, const AbstractBasePtrList &args_spec_list) {
  if (func_graph->manager() == nullptr) {
    std::vector<FuncGraphPtr> graphs{func_graph};
    FuncGraphManagerPtr manager = std::make_shared<FuncGraphManager>(graphs);
    manager->AddFuncGraph(func_graph);
  }
  // Renormalize func_graph to infer and set shape and type information.
  pipeline::ResourcePtr resource_ = std::make_shared<pipeline::Resource>();
  return pipeline::Renormalize(resource_, func_graph, args_spec_list);
}
}  // namespace mindspore
