/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/debug/summary/summary.h"
#include <set>
#include <utility>
#include <memory>
#include "utils/ms_context.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "utils/trace_base.h"

namespace mindspore::debug {
constexpr int kSummaryGetItem = 2;

Summary &Summary::GetInstance() {
  static Summary instance;
  return instance;
}

void Summary::RecurseSetSummaryNodesForAllGraphs(KernelGraph *graph) {
  MS_LOG(INFO) << "Recurse set summary nodes for all graphs in graph: " << graph->graph_id() << " start";
  MS_EXCEPTION_IF_NULL(graph);
  SetSummaryNodes(graph);
  auto &summary_nodes = graph->summary_nodes();
  std::map<std::string, std::pair<AnfNodePtr, int>> summary;
  summary.insert(summary_nodes.cbegin(), summary_nodes.cend());
  auto &child_graphs = graph->child_graph_order();
  for (auto &child_graph : child_graphs) {
    SetSummaryNodes(child_graph.lock().get());
    auto &child_graph_summary = child_graph.lock()->summary_nodes();
    summary.insert(child_graph_summary.cbegin(), child_graph_summary.cend());
    RecurseSetSummaryNodesForAllGraphs(child_graph.lock().get());
  }
  graph->set_summary_nodes(summary);
  MS_LOG(INFO) << "The total summary nodes is: " << summary.size() << " for graph: " << graph->graph_id();
}

void Summary::SummaryTensor(KernelGraph *graph) {
  auto ms_context = MsContext::GetInstance();
  std::string backend = ms_context->backend_policy();
  if (backend == "ge") {
    return;
  }

  if (summary_callback_ == nullptr) {
    return;
  }
  MS_EXCEPTION_IF_NULL(graph);
  bool exist_summary = graph->summary_node_exist();
  if (!exist_summary) {
    return;
  }

  SetSummaryNodes(graph);
  auto summary_outputs = graph->summary_nodes();
  std::map<std::string, tensor::TensorPtr> params_list;
  // fetch outputs apply kernel in session & run callback functions
  for (const auto &output_item : summary_outputs) {
    auto node = output_item.second.first;
    size_t index = IntToSize(output_item.second.second);
    auto address = AnfAlgo::GetOutputAddr(node, index, false);
    auto shape = common::AnfAlgo::GetOutputInferShape(node, index);
    TypeId type_id = common::AnfAlgo::GetOutputInferDataType(node, index);
    tensor::TensorPtr tensor = std::make_shared<tensor::Tensor>(type_id, shape);
    MS_EXCEPTION_IF_NULL(address);
    if (!address->GetPtr()) {
      continue;
    }
    if (!address->SyncDeviceToHost(trans::GetRuntimePaddingShape(node, index), LongToSize(tensor->data().nbytes()),
                                   tensor->data_type(), tensor->data_c())) {
      MS_LOG(ERROR) << "Failed to sync output from device to host.";
    }
    tensor->set_sync_status(kNoNeedSync);
    params_list[output_item.first] = tensor;
  }
  // call callback function here
  summary_callback_(0, params_list);
}

void Summary::RegisterSummaryCallBackFunc(const CallBackFunc &callback) { summary_callback_ = callback; }

void Summary::SetSummaryNodes(KernelGraph *graph) {
  MS_LOG(DEBUG) << "Update summary Start";
  MS_EXCEPTION_IF_NULL(graph);
  if (!graph->summary_node_exist()) {
    return;
  }
  auto summary = graph->summary_nodes();
  auto apply_list = TopoSort(graph->get_return());
  for (auto &n : apply_list) {
    MS_EXCEPTION_IF_NULL(n);
    if (AnfAlgo::IsSummaryNode(n)) {
      auto cnode = n->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (cnode->inputs().size() <= kSummaryGetItem) {
        MS_LOG(EXCEPTION) << "The node Summary should have 2 inputs at least, but got " << (cnode->inputs().size() - 1)
                          << "." << trace::DumpSourceLines(cnode);
      }
      auto node = cnode->input(kSummaryGetItem);
      MS_EXCEPTION_IF_NULL(node);
      auto item_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, false);
      MS_EXCEPTION_IF_NULL(item_with_index.first);
      if (!AnfUtils::IsRealKernel(item_with_index.first)) {
        MS_LOG(EXCEPTION) << "Unexpected node:" << item_with_index.first->DebugString();
      }
      summary[n->fullname_with_scope()] = item_with_index;
    }
  }
  graph->set_summary_nodes(summary);
  MS_LOG(DEBUG) << "Update summary end size: " << summary.size();
}

}  // namespace mindspore::debug
