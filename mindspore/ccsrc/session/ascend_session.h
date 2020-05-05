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
#ifndef MINDSPORE_CCSRC_SESSION_ASCEND_SESSION_H
#define MINDSPORE_CCSRC_SESSION_ASCEND_SESSION_H
#include <unordered_map>
#include <string>
#include <memory>
#include <vector>
#include <utility>
#include <stack>
#include <map>
#include <tuple>
#include <set>
#include "session/session_basic.h"
#include "session/kernel_graph.h"
#include "kernel/kernel.h"
#include "session/session_factory.h"

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
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs) override;
  void BuildGraph(GraphId) override;
  void BuildOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
               const std::vector<tensor::TensorPtr> &input_tensors, const std::vector<bool> &tensors_mask) override;
  py::tuple RunOp(const OpRunInfo &op_run_info, const GraphInfo &graph_info,
                  const std::vector<tensor::TensorPtr> &input_tensors) override;

  // set parameters of final graph
  GraphId SetFinalGraphInput(const std::vector<AnfNodePtr> &args) override;
  // set output of final graph
  void SetFinalGraphOutput(const BaseRef &output) override;
  // insert switch and set the relative active ops
  void SwitchCompile(GraphId cond_g, GraphId true_g, GraphId false_g, const AnfNodePtr &condition_output) override;
  // set args of child graph.the arg maybe come from a output of other child graphs,or from final graph's parameter
  void SetChildGraphInput(GraphId g, const VectorRef &args) override;
  // get graph id in child graphs by ME front anf node pointer
  GraphId GetGraphIdByNode(const AnfNodePtr &front_anf) const override;
  // get graph id of final graph
  GraphId GetFinalRunGraph() const override { return final_graph_id_; }
  // insert active to graph
  void SetActive(GraphId, GraphId) override;
  // compile child graph when session have multiple child graphs
  void CompileChildGraph(const KernelGraphPtr &child_graph);

 private:
  void InitRuntimeResource();
  void SelectKernel(const KernelGraph &kernel_graph) const;
  void HardwareOptimize(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void RunOpAdjustKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void AssignStream(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void BuildKernel(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void MemoryAlloc(KernelGraph *kernel_graph) const;
  void RunOpMemoryAlloc(const std::vector<tensor::TensorPtr> &input_tensors, KernelGraph *kernel_graph) const;
  void GenerateTaskInfo(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void LoadTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void ExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  void Dump(const std::shared_ptr<KernelGraph> &kernel_graph) const;
  // below functions are used for run op
  void RunOpHardwareOptimize(const std::shared_ptr<session::KernelGraph> &kernel_graph) const;
  void RunOpExecTask(const std::shared_ptr<KernelGraph> &kernel_graph) const;

  size_t SetChildGraphInput(const KernelGraphPtr &graph, const AnfNodePtr &node, size_t input_index);
  size_t SetChildGraphInput(const KernelGraphPtr &graph, const ValuePtr &value, size_t input_index);
  size_t SetChildGraphInput(const KernelGraphPtr &graph, const VectorRef &vec_args, size_t input_index);

  // merge execution order list of child graphs
  void MergeGraphExecOrder();
  // insert assion op to sync data bettween different graphs
  void InsertAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to);
  // insert mutiple assigns to graph
  void InsertMultipleAssignToGraph(GraphId graph_id, const AnfNodePtr &from, const AnfNodePtr &to);
  // insert active op to graph
  void InsertStreamActiveToGraph(GraphId graph_id, uint32_t actived_stream);
  // get execute index of graph
  size_t ExecOrderOfChildGraph(GraphId final_graph, GraphId child_graph);
  // handle condition graph from vm
  void InsertSwitchToGraph(GraphId condition_graph_id, GraphId true_graph_id);
  // insert depend to graph, used to attch control nodes to graph
  void InsertDependToGraph(GraphId graph_id, const AnfNodePtr &attch_node);
  // insert depend to graph, used to attch control nodes to graph
  void InsertControlDependToGraph(GraphId graph_id, const AnfNodePtr &first_node, const AnfNodePtr &second_node);
  // Get graph by graph id ,if not exist return null ptr
  KernelGraphPtr GetGraph(GraphId graph_id);
  // set child graph parameter if front arg is a anf
  void SetChildGraphParameter(const AnfNodePtr &front_anf, GraphId to_graph_id, size_t input_idx);
  // set child graph parameter if front arg is a tensor
  void SetChildGraphParameter(const tensor::TensorPtr &front_tensor, GraphId to_graph_id, size_t input_idx);
  // update the execution order of all child graphs
  void UpdateGraphOrder(GraphId to_graph);
  // handle switch when merge
  void MergeSwitchCompile();
  // get graph order vector by graph id
  std::vector<GraphId> &GetGraphOrder(GraphId final_graph_id);
  // get graph order type vector by graph id
  std::vector<GraphType> &GetGraphOrderType(GraphId final_graph_id);
  // copy output of if and else
  void CopyOutputOfIf(GraphId false_graph_id);
  // check if graph cache exist
  bool GraphCacheExist(const GraphInfo &graph_info) const;
  // insert all assign to child graph
  void InsertAllAssigns();
  // create fake output of final graph
  AnfNodePtr CreateFakeOutput(GraphId final_graph_id, const AnfNodePtr &true_output);
  // sync intial tensors' data to device
  void SyncInitialTenosrToDevice();

  // member variables
  // key is final_graph_id,value is child graph execute order of final graph
  std::unordered_map<GraphId, std::vector<GraphId>> graph_execute_orders_;
  // key is final_graph_id,value is the graph types of child graphs
  std::unordered_map<GraphId, std::vector<GraphType>> graph_order_types_;
  // record condition graph of while
  std::unordered_map<GraphId, GraphId> while_condition_graphs_;
  // record all conditions
  std::unordered_map<GraphId, std::pair<GraphId, GraphId>> switches_;
  std::unordered_map<GraphId, AnfNodePtr> condition_output_;
  // share parameters
  std::set<std::tuple<AnfNodePtr, GraphId, size_t>> assigns_;
  // initial tensors, these tensor will sync data to device before run graph
  std::map<std::pair<GraphId, size_t>, tensor::TensorPtr> initial_tenosrs_;
  // final_graph_id is used in every root graph has it's own session situation
  GraphId final_graph_id_;
};
MS_REG_SESSION(kAscendDevice, AscendSession);
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_SESSION_ASCEND_SESSION_H
