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

#include "backend/graph_compiler/backend.h"
#include "backend/graph_compiler/transform.h"
#include "common/device_common_test.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "frontend/operator/ops.h"
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/device/ascend/hal/hardware/ascend_session.h"
#include "plugin/device/ascend/optimizer/mindir/ascend_vm_op_adapter.h"
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

void BackendCommon::PrintGraphNodeList(const FuncGraphPtr &func_graph) {
  std::vector<AnfNodePtr> nodes = TopoSort(func_graph->get_return());
  MS_LOG(INFO) << "======================== " << func_graph->ToString() << " ========================";
  size_t index = 0;
  for (auto &node : nodes) {
    if (node->isa<CNode>() && IsValueNode<Primitive>(node->cast<CNodePtr>()->input(0)) &&
        !IsPrimitiveCNode(node, prim::kPrimReturn)) {
      auto primitive = GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(primitive);
      MS_LOG(INFO) << "Node[" << index << "]:" << node->DebugString() << ", attr text:" << primitive->GetAttrsText();
    } else {
      MS_LOG(INFO) << "Node[" << index << "]:" << node->DebugString();
    }
    index++;
  }
  MS_LOG(INFO) << "======================== graph end ========================";
}

bool BackendCommon::CheckEqualGraph(const FuncGraphPtr &a, const FuncGraphPtr &b) {
  FuncGraphPairMapEquiv equiv_graph_;
  NodeMapEquiv equiv_node_;
  auto ret = Isomorphic(a, b, &equiv_graph_, &equiv_node_);
  if (!ret) {
    MS_LOG(INFO) << "Print Graph infos:";
    PrintGraphNodeList(a);
    PrintGraphNodeList(b);
    MS_LOG(INFO) << "End Graph infos";
  }
  return ret;
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

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device = context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (device == kAscendDevice) {
    auto optimizer = std::make_shared<opt::GraphOptimizer>();
    auto pm = std::make_shared<opt::PassManager>();
    pm->AddPass(std::make_shared<opt::AscendVmOpAdapter>());
    optimizer->AddPassManager(pm);
    optimizer->Optimize(kernel_graph);

    MS_LOG(INFO) << "New Kernel Graph infos:";
    PrintGraphNodeList(kernel_graph);
  }

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
  auto graph = pipeline::Renormalize(resource_, func_graph, args_spec_list);
  MS_LOG(INFO) << "New Function Graph infos:";
  PrintGraphNodeList(graph);
  return graph;
}

std::shared_ptr<session::KernelGraph> BackendCommon::Compile(const FuncGraphPtr &func_graph) {
  auto new_manager = MakeManager({func_graph});
  MS_EXCEPTION_IF_NULL(new_manager);
  new_manager->AddFuncGraph(func_graph);
  func_graph->set_manager(new_manager);

  const std::string kDefaultDeviceName = "CPU";
  auto graph_partition = std::make_shared<compile::GraphPartition>(compile::GetMsNonlinearOps(), kMsConvert);
  bool multi_target = false;
  auto segments = graph_partition->Partition(func_graph, &multi_target);
  if (segments.empty()) {
    return nullptr;
  }
  auto segment = segments[0];
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = compile::TransformSegmentToAnfGraph(segment->nodes_);
  runtime::test::DeviceContextKey device_context_key{kDefaultDeviceName, 0};
  auto device_context = std::make_shared<runtime::test::TestDeviceContext>(device_context_key);

  auto compiler = std::make_shared<compile::GraphCompiler>();
  auto graph_id =
    compiler->CompileGraph(segment, outputs, device_context.get(), device::RunMode::kKernelMode, false);
  return compiler->Fetch(graph_id);
}
}  // namespace mindspore
