/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_H

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <stack>
#include <atomic>
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "utils/contract.h"
#include "runtime/device/kernel_info.h"

namespace mindspore {
namespace session {
using AnfWithOutIndex = std::pair<AnfNodePtr, size_t>;
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
struct KernelWithIndexCmp {
  bool operator()(const KernelWithIndex &key1, const KernelWithIndex &key2) const {
    if (key1.first != key2.first) {
      return key1.first < key2.first;
    }
    if (key1.second != key2.second) {
      return key1.second < key2.second;
    }
    return false;
  }
};

using KernelMapTensor = std::map<session::KernelWithIndex, BaseRef, session::KernelWithIndexCmp>;

class KernelGraph : public FuncGraph {
 public:
  KernelGraph() : graph_id_(0), start_label_(nullptr), end_goto_(nullptr), current_epoch_(0), is_dynamic_shape_(false) {
    inputs_ = std::make_shared<std::vector<AnfNodePtr>>();
    execution_order_ = {};
    mem_reuse_exec_order_ = {};
    executable_ = true;
    summary_node_exist_ = false;
    stream_distinction_label_ = kInvalidDistincLabel;
  }

  KernelGraph(const KernelGraph &graph) : FuncGraph(graph) {
    inputs_ = graph.inputs_;
    child_graph_result_ = graph.child_graph_result_;
    execution_order_ = graph.execution_order_;
    mem_reuse_exec_order_ = graph.mem_reuse_exec_order_;
    graph_id_ = graph.graph_id_;
    stream_distinction_label_ = graph.stream_distinction_label_;
    front_backend_anf_map_ = graph.front_backend_anf_map_;
    backend_front_anf_map_ = graph.backend_front_anf_map_;
    tensor_to_value_node_map_ = graph.tensor_to_value_node_map_;
    graph_value_nodes_ = graph.graph_value_nodes_;
    node_input_num_ = graph.node_input_num_;
    node_input_edges_ = graph.node_input_edges_;
    ref_out_in_map_ = graph.ref_out_in_map_;
    node_output_edges_ = graph.node_output_edges_;
    summary_nodes_ = graph.summary_nodes_;
    updated_parameters_ = graph.updated_parameters_;
    executable_ = graph.executable_;
    summary_node_exist_ = graph.summary_node_exist_;
    valid_inputs_ = graph.valid_inputs_;
    child_graph_order_ = graph.child_graph_order_;
    input_ctrl_tensors_ = graph.input_ctrl_tensors_;
    parent_graph_ = graph.parent_graph_;
    start_label_ = graph.start_label_;
    end_goto_ = graph.end_goto_;
    internal_parameter_to_front_node_map_ = graph.internal_parameter_to_front_node_map_;
    graph_output_to_front_node_map_ = graph.graph_output_to_front_node_map_;
    front_to_internal_outputs_map_ = graph.front_to_internal_outputs_map_;
    internal_outputs_to_front_map_ = graph.internal_outputs_to_front_map_;
    internal_outputs_tensor_map_ = graph.internal_outputs_tensor_map_;
    current_epoch_ = graph.current_epoch_;
    tuple_parameter_to_make_tuple_map_ = graph.tuple_parameter_to_make_tuple_map_;
    visited_nodes_ = graph.visited_nodes_;
    edge_to_ = graph.edge_to_;
    loop_nodes_ = graph.loop_nodes_;
    input_nodes_ = graph.input_nodes_;
    pre_graphs_ = graph.pre_graphs_;
    post_graphs_ = graph.post_graphs_;
    allreduce_from_send_recv_pairs_ = graph.allreduce_from_send_recv_pairs_;
    allreduce_to_send_recv_pairs_ = graph.allreduce_to_send_recv_pairs_;
    size_t pre_graph_finished_count = graph.pre_graph_finished_count_;
    pre_graph_finished_count_ = pre_graph_finished_count;
    size_t post_graph_finished_count = graph.post_graph_finished_count_;
    post_graph_finished_count_ = post_graph_finished_count;
    first_step_ = graph.first_step_;
    has_optimizer_ = graph.has_optimizer_;
    is_dynamic_shape_ = graph.is_dynamic_shape_;
  }

  ~KernelGraph() override;

  MS_DECLARE_PARENT(KernelGraph, FuncGraph);

  const std::vector<AnfNodePtr> &inputs() const;
  std::vector<AnfNodePtr> *MutableInputs() const { return inputs_.get(); }
  void SetGraphInputs(const std::vector<AnfNodePtr> &inputs) {
    inputs_ = std::make_shared<std::vector<AnfNodePtr>>(inputs);
  }
  void ReplaceGraphInput(const AnfNodePtr &old_parameter, const AnfNodePtr &new_parameter);
  std::vector<AnfNodePtr> outputs() const;
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs) override;
  CNodePtr NewCNodeWithInfos(const std::vector<AnfNodePtr> &inputs, const CNodePtr &ori_cnode = nullptr);
  void CreateKernelInfoFromNewParameter(const CNodePtr &cnode);
  CNodePtr NewCNode(const CNodePtr &cnode);
  void ResetAssignInputFeatureMapFlag(const CNodePtr &cnode) const;
  ParameterPtr NewParameter(const ParameterPtr &parameter = nullptr);
  ParameterPtr NewParameter(const abstract::AbstractBasePtr &abstract);
  ValueNodePtr NewValueNode(const AbstractBasePtr &abstract, const ValuePtr &value);
  ValueNodePtr NewValueNode(const ValueNodePtr &value_node = nullptr);
  ValueNodePtr NewValueNode(const tensor::TensorPtr &input_tensor);
  // trans tuple output to maketuple + no_tuple out
  AnfNodePtr TransTupleToMakeTuple(const AnfNodePtr &node);
  void set_execution_order(const std::vector<CNodePtr> &order) { execution_order_ = order; }
  void set_execution_order(std::vector<CNodePtr> &&order) { execution_order_ = std::move(order); }
  const std::vector<CNodePtr> &execution_order() const { return execution_order_; }
  // Set new exec_order for mem_reuse
  void set_mem_reuse_exec_order(const std::vector<CNodePtr> &order) { mem_reuse_exec_order_ = order; }
  const std::vector<CNodePtr> &mem_reuse_exec_order() const { return mem_reuse_exec_order_; }
  void SetExecOrderByDefault();
  uint32_t graph_id() const { return graph_id_; }
  void set_graph_id(uint32_t graph_id) { graph_id_ = graph_id; }
  uint32_t root_graph_id() const { return root_graph_id_; }
  void set_root_graph_id(uint32_t root_graph_id) { root_graph_id_ = root_graph_id; }

  // and a new front to backend anf relation to maop
  void FrontBackendlMapAdd(const AnfNodePtr &front_anf, const AnfNodePtr &backend_anf);
  // replace old backend anf with new backend anf
  void FrontBackendlMapUpdate(const AnfNodePtr &old_backend_anf, const AnfNodePtr &new_backend_anf);
  // get backend anf by front anf
  AnfNodePtr GetBackendAnfByFrontAnf(const AnfNodePtr &front_anf);
  // get front anf by backend anf
  AnfNodePtr GetFrontAnfByBackendAnf(const AnfNodePtr &backend_anf);
  // check backend node whether exist in map
  bool BackendNodeExistInFrontBackendMap(const AnfNodePtr &backend_anf);
  // get value node by tensor
  ValueNodePtr GetValueNodeByTensor(const tensor::TensorPtr &tensor);
  // add value node tensor relation map
  void TensorValueNodeMapAdd(const tensor::TensorPtr &tensor, const ValueNodePtr &value_node);
  // get all value nodes of graph
  const std::unordered_set<ValueNodePtr> graph_value_nodes() const { return graph_value_nodes_; }
  // add value node to graph
  void AddValueNodeToGraph(const ValueNodePtr &value_node);
  // ref output is in map
  bool IsInRefOutputMap(const AnfWithOutIndex &pair) const;
  // get ref correspond pairs
  AnfWithOutIndex GetRefCorrespondOutput(const AnfWithOutIndex &out_pair) const;
  // add ref correspond pairs
  void AddRefCorrespondPairs(const AnfWithOutIndex &final_pair, const AnfWithOutIndex &origin_pair);
  // get map
  std::map<AnfWithOutIndex, AnfWithOutIndex> GetRefMap() const { return ref_out_in_map_; }
  // check whether graph is executable
  bool executable() const { return executable_; }
  // set executable of graph
  void set_executable(bool executable) { executable_ = executable; }
#ifndef ENABLE_SECURITY
  // set summary_node of graph
  void set_summary_node_exist(bool summary_node_exist) { summary_node_exist_ = summary_node_exist; }
#endif
  // check whether exist summary node in graph
  bool summary_node_exist() const { return summary_node_exist_; }
  // set invalid inputs for control sink
  std::vector<bool> *MutableValidInputs() { return &valid_inputs_; }
  std::vector<bool> valid_inputs() const { return valid_inputs_; }
  // replace node in graph
  void ReplaceNode(const AnfNodePtr &old_anf_node, const AnfNodePtr &new_anf_node);
  // set stream label of graph
  void set_stream_distinction_label(uint32_t stream_label) { stream_distinction_label_ = stream_label; }
  // get stream label of graph
  uint32_t stream_distinction_label() { return stream_distinction_label_; }
  // refresh execute kernel stream label
  void UpdateExecuteKernelStreamLabel();
  // calculate the leaf graph order of root graph
  std::vector<std::shared_ptr<KernelGraph>> GetLeafGraphOrder();
  // the child graph of current graph
  const std::vector<std::weak_ptr<KernelGraph>> &child_graph_order() const { return child_graph_order_; }
  void set_child_graph_order(const std::vector<std::weak_ptr<KernelGraph>> &order) { child_graph_order_ = order; }
  // checkout whether current graph is leaf graph
  bool IsLeafGraph() const;

  // set input_tensors pointer of control parameter
  void set_input_ctrl_tensors(const std::shared_ptr<std::vector<tensor::TensorPtr>> &input_tensors_ptr) {
    input_ctrl_tensors_ = input_tensors_ptr;
  }
  // get input_tensors pointer of control parameter
  std::shared_ptr<std::vector<tensor::TensorPtr>> input_ctrl_tensors() const { return input_ctrl_tensors_; }
  // get parent kernel graph
  std::weak_ptr<KernelGraph> parent_graph() const { return parent_graph_; }
  // set parent kernel graph
  void set_parent_graph(const std::weak_ptr<KernelGraph> &parent_graph) { parent_graph_ = parent_graph; }
  // find anf node in graph
  std::vector<CNodePtr> FindNodeByPrimitive(const PrimitivePtr &primitive) const;
  std::vector<CNodePtr> FindNodeByPrimitive(const std::vector<PrimitivePtr> &primitive_list) const;
  // used to dump ir
  std::string ToString() const override;

  void set_start_label(const CNodePtr &start_label) { start_label_ = start_label; }
  CNodePtr get_start_label() { return start_label_; }
  void set_end_goto(const CNodePtr &end_goto) { end_goto_ = end_goto; }
  CNodePtr get_end_goto() { return end_goto_; }
  void PrintGraphExecuteOrder() const;
  const std::map<std::string, std::pair<AnfNodePtr, int>> &summary_nodes() const { return summary_nodes_; }
  void set_summary_nodes(const std::map<std::string, std::pair<AnfNodePtr, int>> &nodes) { summary_nodes_ = nodes; }
  void AddInternalOutput(const AnfNodePtr &front_node, const AnfNodePtr &node, size_t output_idx, bool unique_target);
  void ReplaceInternalOutput(const AnfNodePtr &node, const AnfNodePtr &new_node, size_t src_output_idx,
                             size_t dst_output_idx);
  void ReplaceInternalOutput(const AnfNodePtr &node, const AnfNodePtr &new_node);
  AnfNodePtr GetInternalOutputByFrontNode(const AnfNodePtr &front_node) const;
  bool IsInternalOutput(const AnfNodePtr &node, size_t output_idx) const;
  bool IsInternalOutput(const AnfNodePtr &node) const;
  bool IsUniqueTargetInternalOutput(const AnfNodePtr &node, size_t output_idx) const;
  void AddInternalOutputTensor(const AnfNodePtr &node, size_t output_idx, const tensor::TensorPtr &tensor);
  tensor::TensorPtr GetInternalOutputTensor(const AnfNodePtr &node, size_t output_idx);

  // Cache the internal parameter and corresponding to front node into internal_parameter_to_front_node_map_.
  void CacheInternalParameterToFrontNode(const AnfNodePtr &parameter, const AnfWithOutIndex &front_node_with_index);
  AnfWithOutIndex GetFrontNodeByInternalParameter(const AnfNodePtr &parameter) const;

  // Get the funcgraph to which the kernel graph belongs.
  FuncGraphPtr GetFuncGraph();
  // Cache the backend graph output nodes and corresponding to front nodes with output index into
  // graph_output_to_front_node_map_.
  void CacheGraphOutputToFrontNodeWithIndex(const AnfNodePtr &backend_graph_output, const AnfNodePtr &front_node);
  AnfWithOutIndex GetFrontNodeWithIndexByGraphOutput(const AnfWithOutIndex &backend_graph_output_with_index) const;
  // Update the related map of backend graph output nodes by modified backend output nodes.
  void UpdateGraphOutputMap(const std::vector<AnfWithOutIndex> &old_outputs,
                            const std::vector<AnfWithOutIndex> &new_outputs);

  uint32_t current_epoch() const { return current_epoch_; }
  void set_current_epoch(uint32_t epoch) { current_epoch_ = epoch; }
  void UpdateChildGraphOrder();
  const std::vector<AnfNodePtr> &child_graph_result() const { return child_graph_result_; }
  void AddChildGraphResult(const AnfNodePtr &parameter) { child_graph_result_.push_back(parameter); }
  void set_child_graph_result(const std::vector<AnfNodePtr> &child_graph_result) {
    child_graph_result_ = child_graph_result;
  }

  void InsertTupleParameterToMakeTupleMap(const AnfNodePtr &param, const AnfNodePtr &make_tuple) {
    if (tuple_parameter_to_make_tuple_map_.find(param) != tuple_parameter_to_make_tuple_map_.end()) {
      return;
    }
    tuple_parameter_to_make_tuple_map_[param] = make_tuple;
  }
  AnfNodePtr FindTupleParameterToMakeTupleMap(const AnfNodePtr &param) {
    if (tuple_parameter_to_make_tuple_map_.find(param) != tuple_parameter_to_make_tuple_map_.end()) {
      return tuple_parameter_to_make_tuple_map_[param];
    } else {
      return nullptr;
    }
  }
  void RemoveNodeFromGraph(const AnfNodePtr &node);
  void UpdateGraphDynamicAttr();
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  void SetOptimizerFlag();
  void SetInputNodes();
  const std::vector<AnfNodePtr> &input_nodes() const { return input_nodes_; }
  void SetInputTensors(const std::vector<tensor::TensorPtr> &input_tensors) { input_tensors_ = input_tensors; }
  const std::vector<tensor::TensorPtr> &input_tensors() const { return input_tensors_; }

  void SetOutputNodeToTensor(const KernelMapTensor &node_to_tensor) { output_node_to_tensor_ = node_to_tensor; }

  tensor::TensorPtr GetNodeOutputTensor(const session::KernelWithIndex &output_index) const {
    auto iter = output_node_to_tensor_.find(output_index);
    if (iter != output_node_to_tensor_.end()) {
      return utils::cast<tensor::TensorPtr>(iter->second);
    }
    return nullptr;
  }

  bool has_optimizer() const { return has_optimizer_; }
  bool IsUpdatedParameter(const ParameterPtr &param) const {
    if (updated_parameters_.find(param) != updated_parameters_.end()) {
      return true;
    }
    return false;
  }
  // handle graph dependency
  void AddPreGraph(const std::shared_ptr<session::KernelGraph> &graph) {
    if (graph != nullptr) {
      pre_graphs_[graph->graph_id()] = graph;
    }
  }
  void AddPostGraph(const std::shared_ptr<session::KernelGraph> &graph) {
    if (graph != nullptr) {
      post_graphs_[graph->graph_id()] = graph;
    }
  }

  bool IsPreGraphFinished() const { return pre_graphs_.size() == pre_graph_finished_count_; }
  bool IsPostGraphFinished() const {
    if (first_step_) {
      return true;
    }
    return post_graphs_.size() == post_graph_finished_count_;
  }

  bool HasPostGraph() const { return !post_graphs_.empty(); }

  void IncPreGraphFinishedCount() { pre_graph_finished_count_++; }
  void IncPostGraphFinishedCount() { post_graph_finished_count_++; }
  void ResetGraphRunningStatus() {
    first_step_ = false;
    post_graph_finished_count_ = 0;
    pre_graph_finished_count_ = 0;
  }
  void OnRunGraphFinished() {
    for (auto post_graph : post_graphs_) {
      auto post_graph_ptr = post_graph.second.lock();
      if (post_graph_ptr != nullptr) {
        post_graph_ptr->IncPreGraphFinishedCount();
      }
    }
    for (auto pre_graph : pre_graphs_) {
      auto pre_graph_ptr = pre_graph.second.lock();
      if (pre_graph_ptr != nullptr) {
        pre_graph_ptr->IncPostGraphFinishedCount();
      }
    }
  }
  // end of handle graph dependency

  // The interface of allreduce send/recv pairs map.
  void InsertFromSendRecvPair(const CNodePtr &allreduce, const std::pair<CNodePtr, CNodePtr> &send_recv_pair) {
    allreduce_from_send_recv_pairs_[allreduce] = send_recv_pair;
  }
  void InsertToSendRecvPair(const CNodePtr &allreduce, const std::pair<CNodePtr, CNodePtr> &send_recv_pair) {
    allreduce_to_send_recv_pairs_[allreduce] = send_recv_pair;
  }
  const std::unordered_map<CNodePtr, std::pair<CNodePtr, CNodePtr>> &allreduce_from_send_recv_pairs() const {
    return allreduce_from_send_recv_pairs_;
  }
  const std::unordered_map<CNodePtr, std::pair<CNodePtr, CNodePtr>> &allreduce_to_send_recv_pairs() const {
    return allreduce_to_send_recv_pairs_;
  }

  uint32_t label_num() const { return label_num_; }
  void set_label_num(uint32_t num) { label_num_ = num; }
  // The graphs has recursion.
  bool recursive_call() const { return has_recursive_call_; }
  // The graphs has subgraph multi-call.
  bool subgraph_multi_call() const { return has_subgraph_multicall_; }
  // set flag to indicate whether has recursion.
  void set_recursive_call(bool flag) { has_recursive_call_ = flag; }
  // set flag to indicate whether has multi-call.
  void set_subgraph_multi_call(bool flag) { has_subgraph_multicall_ = flag; }

  bool is_all_nop_node() const { return is_all_nop_node_; }
  void set_is_all_nop_node(bool is_all_nop_node) { is_all_nop_node_ = is_all_nop_node; }
  std::map<AnfWithOutIndex, AnfWithOutIndex> graph_output_map() { return graph_output_to_front_node_map_; }

  // The interface to set/get the graph GIL flag.
  void set_is_need_gil(bool flag) { is_need_gil_ = flag; }
  bool is_need_gil() { return is_need_gil_; }

  bool IsDatasetGraph() const;

 private:
  // remove value node form graph
  bool RemoveValueNodeFromGraph(const ValueNodePtr &value_node);
  void SetKernelInfoForNode(const AnfNodePtr &node) const;
  AnfNodePtr MakeValueNode(const AnfNodePtr &node) const;
  void EnqueueActiveNodes(const AnfNodePtr &node, std::queue<AnfNodePtr> *visit_queue,
                          std::unordered_set<AnfNodePtr> *visited_nodes, bool comm_first = true);
  // update node edge list
  void UpdateNodeEdgeList(std::queue<AnfNodePtr> *seed_nodes);
  // add node depend edge by data edge
  void AddDependEdge(const AnfNodePtr &node, const AnfNodePtr &input, size_t depend_edge_num);
  std::vector<AnfNodePtr> GetOutputNodes(const AnfNodePtr &node);
  AnfNodePtr TransValueNodeTuple(const AbstractBasePtr &abstract, const ValuePtr &value);
  AnfNodePtr TransParameterTuple(const AbstractBasePtr &abstract);
  AnfNodePtr TransCNodeTuple(const CNodePtr &node);
  AnfNodePtr CreatTupleGetItemNode(const AnfNodePtr &node, size_t output_idx);
  std::vector<CNodePtr> SortStartLabelAndEndGoto();
  // checkout whether loop exist in graph
  void CheckLoop();
  uint32_t GetLoopNum(const std::map<AnfNodePtr, size_t> &none_zero_nodes);
  void GetLoopNodesByDFS(const AnfNodePtr &node, uint32_t *loop_num);

  // members
  std::shared_ptr<std::vector<AnfNodePtr>> inputs_;
  std::vector<AnfNodePtr> child_graph_result_;
  std::vector<CNodePtr> execution_order_;
  std::vector<CNodePtr> mem_reuse_exec_order_;
  uint32_t graph_id_;
  uint32_t stream_distinction_label_;
  uint32_t root_graph_id_{0};

  // record map bettween front anf and backend anf,use two map implement bidirectional map
  std::unordered_map<AnfNodePtr, AnfNodePtr> front_backend_anf_map_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> backend_front_anf_map_;
  // there may be a tensor from ME backend ,a value ndoe will be create according the tensor,map record
  std::unordered_map<tensor::TensorPtr, ValueNodePtr> tensor_to_value_node_map_;
  // include all value nodes
  std::unordered_set<ValueNodePtr> graph_value_nodes_;
  std::unordered_map<AnfNodePtr, size_t> node_input_num_;
  std::unordered_map<AnfNodePtr, std::vector<std::pair<AnfNodePtr, size_t>>> node_input_edges_;
  // record map between ref final output anf with index and ref origin input with index
  std::map<AnfWithOutIndex, AnfWithOutIndex> ref_out_in_map_;
  std::unordered_map<AnfNodePtr, std::vector<std::pair<AnfNodePtr, size_t>>> node_output_edges_;
  std::map<std::string, std::pair<AnfNodePtr, int>> summary_nodes_;
  // parameters that will be updated when graph is executed
  std::unordered_set<ParameterPtr> updated_parameters_;
  // graph needn't execute
  bool executable_{false};
  // exist summary node in graph
  bool summary_node_exist_{false};
  // valid inputs
  std::vector<bool> valid_inputs_;

  // child graph execute order in parent graph
  std::vector<std::weak_ptr<KernelGraph>> child_graph_order_;

  // input_tensors of control parameter
  std::shared_ptr<std::vector<tensor::TensorPtr>> input_ctrl_tensors_;

  // parameter graph
  std::weak_ptr<KernelGraph> parent_graph_;

  CNodePtr start_label_;
  CNodePtr end_goto_;

  // Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
  // related to the input of this kernel graph. The first of unordered map is the input of this kernel graph, the second
  // of unordered map is front node corresponding to the output of previous kernel graph.
  std::unordered_map<AnfNodePtr, AnfWithOutIndex> internal_parameter_to_front_node_map_;
  // The first of map is the backend graph output of this kernel graph, the second of map is front node corresponding to
  // the backend node with index.
  std::map<AnfWithOutIndex, AnfWithOutIndex> graph_output_to_front_node_map_;

  std::unordered_map<AnfNodePtr, AnfNodePtr> front_to_internal_outputs_map_;
  std::unordered_map<AnfNodePtr, std::unordered_map<size_t, std::pair<AnfNodePtr, bool>>>
    internal_outputs_to_front_map_;
  std::unordered_map<AnfNodePtr, std::unordered_map<size_t, tensor::TensorPtr>> internal_outputs_tensor_map_;
  uint32_t current_epoch_;
  std::unordered_map<AnfNodePtr, AnfNodePtr> tuple_parameter_to_make_tuple_map_;
  std::set<AnfNodePtr> visited_nodes_;
  std::map<AnfNodePtr, AnfNodePtr> edge_to_;
  std::stack<AnfNodePtr> loop_nodes_;
  std::vector<AnfNodePtr> input_nodes_;
  std::vector<tensor::TensorPtr> input_tensors_;
  KernelMapTensor output_node_to_tensor_;
  std::unordered_map<uint32_t, std::weak_ptr<session::KernelGraph>> pre_graphs_;
  std::unordered_map<uint32_t, std::weak_ptr<session::KernelGraph>> post_graphs_;
  // The send/recv pairs inserted for allreduce, the key is allreduce kernel, the first of pair is send node, the second
  // of pair is recv node.
  std::unordered_map<CNodePtr, std::pair<CNodePtr, CNodePtr>> allreduce_from_send_recv_pairs_;
  std::unordered_map<CNodePtr, std::pair<CNodePtr, CNodePtr>> allreduce_to_send_recv_pairs_;
  std::atomic<size_t> pre_graph_finished_count_{0};
  std::atomic<size_t> post_graph_finished_count_{0};
  bool first_step_{true};
  bool has_optimizer_{false};
  bool is_dynamic_shape_{false};

  // Indicate the graphs has recursion or multi-call or not as the root graph.
  bool has_recursive_call_{false};
  bool has_subgraph_multicall_{false};

  // Number of labels. This is also the 'batch_num' for DavinciModel,
  // It should be 1 if no labels used for control flow.
  uint32_t label_num_ = 1;

  // If all the nodes of graph is the nop node.
  bool is_all_nop_node_{false};

  // Indicate whether the kernels in the graphs acquire Python GIL.
  bool is_need_gil_{false};
};
}  // namespace session
using KernelGraphPtr = std::shared_ptr<session::KernelGraph>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_H
