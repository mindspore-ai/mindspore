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

namespace mindspore {
namespace session {
enum GraphType : int { COMMON_GRAPH = 0, CONDITION_GRAPH = 1, BRANCH_START = 2, BRANCH_END = 3 };

class AscendSession : public SessionBasic {
 public:
  AscendSession() { final_graph_id_ = kInvalidGraphId; }
  ~AscendSession() override = default;
  void Init(uint32_t device_id) override {
    SessionBasic::Init(device_id);
    context_ = std::make_shared<Context>(kAscendDevice, device_id);
  }
  GraphId CompileGraph(const AnfNodePtrList &lst, const AnfNodePtrList &outputs) override;
  GraphId CompileGraph(NotNull<FuncGraphPtr> func_graph) override;
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  void BuildGraph(GraphId) override;
  void BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
               const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<int> &tensors_mask) override;
  py::tuple RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                  const std::vector<tensor::TensorPtr> &input_tensors) override;

  // get graph id in child graphs by ME front anf node pointer
  GraphId GetGraphIdByNode(const AnfNodePtr &front_anf) const override;
  // get graph id of final graph
  GraphId GetFinalRunGraph() const override { return final_graph_id_; }
  // compile child graph when session have multiple child graphs
  void CompileChildGraph(const KernelGraphPtr &child_graph);

 private:
  void RecurseSetSummaryNodes(KernelGraph *graph, std::map<std::string, std::pair<AnfNodePtr, int>> *summary);
  void SetSummaryNodes(KernelGraph *graph) override;
  void InitRuntimeResource();
  void SelectKernel(const KernelGraph &kernel_graph) const;
  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AssignStream(NotNull<KernelGraphPtr> kernel_graph) const;
  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void MemoryAlloc(KernelGraph *kernel_graph) const;
  void RunOpMemoryAlloc(const ValuePtr &pre_output_value, const std::vector<tensor::TensorPtr> &input_tensors,
                        KernelGraph *kernel_graph) const;
  void RunOpMemoryClear(const KernelGraph *kernel_graph) const;
  void GenerateTaskInfo(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void LoadTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void ExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void DumpAllGraphs(const std::vector<KernelGraphPtr> &all_graphs);
  void LoadTensor(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  // below functions are used for run op
  void RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const;
  void RunOpExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  static void BackendOptimization(const std::vector<KernelGraphPtr> &all_graphs);
  static void LinkChildGraphs(NotNull<KernelGraphPtr> graph);
  void RootGraphExecutorValidate(NotNull<KernelGraphPtr> graph);
  // merge execution order list of child graphs
  void MergeGraphExecOrder();
  // insert assion op to sync data bettween different graphs
  void InsertAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to);
  // get graph order vector by graph id
  const std::vector<GraphId> &GetGraphOrder(GraphId final_graph_id) const;
  // get graph order type vector by graph id
  const std::vector<GraphType> &GetGraphOrderType(GraphId final_graph_id) const;
  // check if graph cache exist
  bool GraphCacheExist(const GraphInfo &graph_info) const;
  // insert all assign to child graph
  void InsertAllAssigns();
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
  void AssignStaticMemory(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;
  void UpdateRefOutputMap(const NotNull<KernelGraphPtr> graph, NotNull<std::set<KernelGraphPtr> *> memo) const;

  // key is final_graph_id,value is child graph execute order of final graph
  std::unordered_map<GraphId, std::vector<GraphId>> graph_execute_orders_;
  // key is final_graph_id,value is the graph types of child graphs
  std::unordered_map<GraphId, std::vector<GraphType>> graph_order_types_;
  // share parameters
  std::vector<std::tuple<AnfNodePtr, GraphId, size_t>> assigns_;
  // initial tensors, these tensor will sync data to device before run graph
  std::map<std::pair<GraphId, size_t>, tensor::TensorPtr> initial_tenosrs_;
  // final_graph_id is used in every root graph has it's own session situation
  GraphId final_graph_id_;
};
MS_REG_SESSION(kAscendDevice, AscendSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_ASCEND_SESSION_H
