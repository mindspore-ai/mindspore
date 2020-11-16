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

#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H

#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <stack>
#include <map>
#include <tuple>
#include <set>
#include "backend/session/session_basic.h"
#include "backend/session/kernel_graph.h"
#include "backend/kernel_compiler/kernel.h"
#include "backend/session/session_factory.h"
#include "backend/session/ascend_control_parser.h"
#include "runtime/context.h"

namespace mindspore {
namespace session {
enum GraphType : int { COMMON_GRAPH = 0, CONDITION_GRAPH = 1, BRANCH_START = 2, BRANCH_END = 3 };

class AscendSession : public SessionBasic {
 public:
  AscendSession() { final_graph_id_ = kInvalidGraphId; }
  ~AscendSession() {
    if (rt_context_ != nullptr) {
      auto ret = rtCtxDestroy(rt_context_);
      if (ret != RT_ERROR_NONE) {
        MS_EXCEPTION(DeviceProcessError) << "Call rtCtxDestroy, ret[" << ret << "]";
      }
      rt_context_ = nullptr;
    }
  }

  void Init(uint32_t device_id) override {
    InitDevice(kAscendDevice, device_id);
    auto ret = rtCtxCreate(&rt_context_, 0, device_id);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxCreate, ret[" << static_cast<int>(ret) << "]";
    }
    ret = rtCtxSetCurrent(rt_context_);
    if (ret != RT_ERROR_NONE) {
      MS_EXCEPTION(DeviceProcessError) << "Call rtCtxSetCurrent, ret[" << ret << "]";
    }
  }

  // get graph id in child graphs by ME front anf node pointer
  GraphId GetGraphIdByNode(const AnfNodePtr &front_anf) const override;
  // get graph id of final graph
  GraphId GetFinalRunGraph() const override { return final_graph_id_; }

 protected:
  GraphId CompileGraphImpl(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph) override;
  GraphId CompileGraphImpl(NotNull<FuncGraphPtr> func_graph, const std::vector<tensor::TensorPtr> &inputs) override;
  bool IsSupportSummary() override;
  void RunGraphImpl(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  void BuildGraphImpl(GraphId) override;
  void BuildOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                   const std::vector<tensor::TensorPtr> &input_tensors,
                   const std::vector<int64_t> &tensors_mask) override;
  void RunOpImpl(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                 const std::vector<tensor::TensorPtr> &input_tensors, VectorRef *outputs) override;

 private:
  // compile child graph when session have multiple child graphs
  void CompileChildGraph(const KernelGraphPtr &child_graph);
  void RecurseSetSummaryNodes(KernelGraph *graph, std::map<std::string, std::pair<AnfNodePtr, int>> *summary);
  void SetSummaryNodes(KernelGraph *graph) override;
  void InitRuntimeResource();
  void SelectKernel(const KernelGraph &kernel_graph) const;
  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void GraphKernelOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AssignStream(NotNull<KernelGraphPtr> kernel_graph) const;
  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void BuildDynamicKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void MemoryAlloc(KernelGraph *kernel_graph) const;
  void RunOpMemoryAlloc(const ValuePtr &pre_output_value, const std::vector<tensor::TensorPtr> &input_tensors,
                        KernelGraph *kernel_graph) const;
  void RunOpMemoryClear(const KernelGraph *kernel_graph) const;
  void Load(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void Execute(const std::shared_ptr<KernelGraph> &kernel_graph, bool is_task) const;
  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void DumpAllGraphs(const std::vector<KernelGraphPtr> &all_graphs);
  void LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  // below functions are used for run op
  void RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const;

  static void BackendOptimization(const std::vector<KernelGraphPtr> &all_graphs);
  static void LinkChildGraphs(NotNull<KernelGraphPtr> graph);
  void RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph);
  // merge execution order list of child graphs
  void MergeGraphExecOrder();
  // get graph order vector by graph id
  const std::vector<GraphId> &GetGraphOrder(GraphId final_graph_id) const;
  // get graph order type vector by graph id
  const std::vector<GraphType> &GetGraphOrderType(GraphId final_graph_id) const;
  // check if graph cache exist
  bool GraphCacheExist(const GraphInfo &graph_info) const;
  // sync intial tensors' data to device
  void SyncInitialTenosrToDevice();
  void SetFinalGraphSummaryFlag(const std::shared_ptr<KernelGraph> &kernel_graph);
  // create parameter to receive data from multiple branch output
  void CreateMultiBranchOutput(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo);
  void SelectKernel(NotNull<KernelGraphPtr> root_graph);
  void RecurseSelectKernelInfo(NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> const memo,
                               size_t *const raise_precision_count, size_t *const reduce_precision_count) const;
  void IrFusionPass(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo);
  void HardwareOptimize(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void LoadGraphsToDbg(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void AssignStaticMemory(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void UpdateRefOutputMap(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;

  // key is final_graph_id,value is child graph execute order of final graph
  std::unordered_map<GraphId, std::vector<GraphId>> graph_execute_orders_;
  // key is final_graph_id,value is the graph types of child graphs
  std::unordered_map<GraphId, std::vector<GraphType>> graph_order_types_;
  // initial tensors, these tensor will sync data to device before run graph
  std::map<std::pair<GraphId, size_t>, tensor::TensorPtr> initial_tenosrs_;
  // final_graph_id is used in every root graph has it's own session situation
  GraphId final_graph_id_;
  // ascend runtime context
  rtContext_t rt_context_{nullptr};
};
MS_REG_SESSION(kAscendDevice, AscendSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H
