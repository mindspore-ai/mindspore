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
#include "cxx_api/model/acl/acl_vm/acl_multi_graph_session.h"
#include <memory>
#include <deque>
#include <vector>
#include "backend/common/session/session_factory.h"
#include "include/backend/optimizer/optimizer.h"
#ifdef ENABLE_D
#include "runtime/hardware/device_context_manager.h"
#else
#include "plugin/device/ascend/optimizer/enhancer/add_placeholder_for_dynamic_rnn.h"
#endif
#include "cxx_api/model/acl/model_converter.h"
#include "cxx_api/model/acl/acl_model_options.h"
#include "cxx_api/model/acl/acl_vm/ms_tensor_ref.h"
#include "cxx_api/graph/graph_data.h"

namespace mindspore::session {
void MultiGraphAclSession::Init(uint32_t device_id) { InitExecutor(kDavinciMultiGraphInferenceDevice, device_id); }

GraphId MultiGraphAclSession::CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) {
  class FirstGraphModeGuard {
   public:
    explicit FirstGraphModeGuard(const std::shared_ptr<AclModelOptions> &options) : options_(options) {
      if (options_ != nullptr) {
        options_->SetFirstGraph(true);
      }
    }
    ~FirstGraphModeGuard() {
      if (options_ != nullptr) {
        options_->SetFirstGraph(false);
      }
    }

   private:
    std::shared_ptr<AclModelOptions> options_;
  };
  MS_LOG(INFO) << "Start MultiGraph Compile.";
  // construct kernel graph
  auto kernel_graph = SessionBasic::ConstructKernelGraph(lst, outputs, device::DeviceType::kUnknown, false);
  MS_EXCEPTION_IF_NULL(kernel_graph);
#ifdef ENABLE_D
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context);
  auto deprecated_ptr = device_context->GetDeprecatedInterface();
  MS_EXCEPTION_IF_NULL(deprecated_ptr);
  deprecated_ptr->AclOptimizer(kernel_graph);
#else
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("310_multi_graph_pm");
  pm->AddPass(std::make_shared<opt::InsertPlaceholderForDynamicRNN>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
#endif
  kernel_graph->SetExecOrderByDefault();
  // concert to om data
  ModelConverter model_converter_;
  model_converter_.set_options(options_);
  FirstGraphModeGuard guard(options_);
  auto om_data = model_converter_.LoadMindIR(kernel_graph);
  if (om_data.Data() == nullptr || om_data.DataSize() == 0) {
    MS_LOG(EXCEPTION) << "Load MindIR failed.";
  }
  // load
  std::shared_ptr<Graph> graph = std::make_shared<Graph>(std::make_shared<Graph::GraphData>(om_data, ModelType::kOM));
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_cell = GraphCell(graph);
  auto ret = graph_cell.Load(options_->GetDeviceID());
  if (ret != kSuccess) {
    MS_LOG(EXCEPTION) << "Load failed.";
  }
  graph_cells_[kernel_graph->graph_id()] = graph_cell;
  kernel_graphs_[kernel_graph->graph_id()] = kernel_graph;
  MS_LOG(INFO) << "Multi graph compile success, graph id " << kernel_graph->graph_id();
  return kernel_graph->graph_id();
}

void MultiGraphAclSession::RunGraph(GraphId graph_id, const std::vector<MSTensor> &inputs, VectorRef *outputs) {
  MS_EXCEPTION_IF_NULL(outputs);
  MS_LOG(INFO) << "Start run graph " << graph_id;
  auto iter = graph_cells_.find(graph_id);
  if (iter == graph_cells_.cend()) {
    MS_LOG(EXCEPTION) << "Graph id " << graph_id << " not found.";
  }
  std::vector<MSTensor> out_tensors;
  auto ret = iter->second.Run(inputs, &out_tensors);
  if (ret != kSuccess) {
    MS_LOG(EXCEPTION) << "Graph id " << graph_id << " run failed.";
  }

  std::deque<MSTensor> out_tensors_deque(out_tensors.begin(), out_tensors.end());
  (*outputs) = ConstructOutputRef(graph_id, &out_tensors_deque);
}

VectorRef MultiGraphAclSession::ConstructOutputRef(GraphId graph_id, std::deque<MSTensor> *out_tensors) {
  MS_EXCEPTION_IF_NULL(out_tensors);
  VectorRef outs;
  auto out_nodes = kernel_graphs_[graph_id]->outputs();
  for (auto &out : out_nodes) {
    auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(
      out, 0, false, std::vector<PrimitivePtr>{prim::kPrimMakeTuple, prim::kPrimUpdateState, prim::kPrimStateSetItem});
    auto &anf_node = item_with_index.first;
    if (common::AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimMakeTuple)) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      outs.emplace_back(ConstructOutputRefByTupleNode(cnode, out_tensors));
    } else if (AnfUtils::IsRealKernel(anf_node)) {
      if (out_tensors->empty()) {
        MS_LOG(EXCEPTION) << "Can not find MSTensor for output node " << out->DebugString()
                          << ", visited: " << anf_node->DebugString();
      }
      outs.emplace_back(MSTensorRef(out_tensors->front()));
      out_tensors->pop_front();
    }
  }

  if (!out_tensors->empty()) {
    MS_LOG(EXCEPTION) << "Number of output size " << outs.size() << " but " << out_tensors->size()
                      << " MSTensor remained.";
  }

  return outs;
}

VectorRef MultiGraphAclSession::ConstructOutputRefByTupleNode(const CNodePtr &tuple_node,
                                                              std::deque<MSTensor> *out_tensors) {
  MS_EXCEPTION_IF_NULL(out_tensors);
  VectorRef outs;
  for (size_t i = 1; i < tuple_node->inputs().size(); ++i) {
    auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(
      tuple_node->input(i), 0, false,
      std::vector<PrimitivePtr>{prim::kPrimMakeTuple, prim::kPrimUpdateState, prim::kPrimStateSetItem});
    auto &anf_node = item_with_index.first;
    if (common::AnfAlgo::CheckPrimitiveType(anf_node, prim::kPrimMakeTuple)) {
      auto cnode = anf_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      outs.emplace_back(ConstructOutputRefByTupleNode(cnode, out_tensors));
    } else if (AnfUtils::IsRealKernel(anf_node)) {
      if (out_tensors->empty()) {
        MS_LOG(EXCEPTION) << "Can not find MSTensor for output node " << tuple_node->input(i)->DebugString()
                          << ", visited: " << anf_node->DebugString();
      }
      outs.emplace_back(MSTensorRef(out_tensors->front()));
      out_tensors->pop_front();
    }
  }

  return outs;
}
MS_REG_SESSION(kDavinciMultiGraphInferenceDevice, MultiGraphAclSession);
}  // namespace mindspore::session
