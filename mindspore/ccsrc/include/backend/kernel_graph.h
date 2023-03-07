/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include <set>
#include <stack>
#include <atomic>
#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "ir/func_graph.h"
#include "ir/anf.h"
#include "ir/graph_utils.h"
#include "include/common/utils/contract.h"
#include "include/backend/device_type.h"
#include "include/backend/kernel_info.h"
#include "include/backend/visible.h"

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

struct SomasInfo {
  // whole_block_size_ is 0 indicating that somas did not allocate memory for this graph.
  size_t whole_block_size_{0};
  // offset -> aligned_size_
  std::map<size_t, size_t> merged_blocks_map_;

  // Alloc the base address of graph during execution, which is variable.
  void *base_address_{nullptr};
  // Block offset -> address.
  std::map<size_t, void *> merged_base_addresses_;

  // The owner graph id.
  uint32_t graph_id_{0};
};

using DeviceType = device::DeviceType;
using KernelMapTensor = std::map<session::KernelWithIndex, BaseRef, session::KernelWithIndexCmp>;

class BACKEND_EXPORT KernelGraph : public FuncGraph {
 public:
  KernelGraph()
      : inputs_(std::make_shared<std::vector<AnfNodePtr>>()),
        somas_info_(std::make_shared<SomasInfo>()),
        graph_id_(0),
        stream_distinction_label_(kInvalidDistincLabel),
        device_target_(DeviceType::kUnknown),
        executable_(true),
        summary_node_exist_(false),
        need_inline_(false),
        start_label_(nullptr),
        end_goto_(nullptr),
        current_epoch_(0),
        is_dynamic_shape_(false) {}

  KernelGraph(const KernelGraph &graph) : FuncGraph(graph) {
    inputs_ = graph.inputs_;
    somas_info_ = graph.somas_info_;
    child_graph_result_ = graph.child_graph_result_;
    execution_order_ = graph.execution_order_;
    mem_reuse_exec_order_ = graph.mem_reuse_exec_order_;
    graph_id_ = graph.graph_id_;
    device_target_ = graph.device_target_;
    stream_distinction_label_ = graph.stream_distinction_label_;
    front_backend_anf_map_ = graph.front_backend_anf_map_;
    backend_front_anf_map_ = graph.backend_front_anf_map_;
    tensor_to_value_node_map_ = graph.tensor_to_value_node_map_;
    graph_value_nodes_ = graph.graph_value_nodes_;
    ref_out_in_map_ = graph.ref_out_in_map_;
    node_output_edges_ = graph.node_output_edges_;
    summary_nodes_ = graph.summary_nodes_;
    updated_parameters_ = graph.updated_parameters_;
    executable_ = graph.executable_;
    summary_node_exist_ = graph.summary_node_exist_;
    need_inline_ = graph.need_inline_;
    valid_inputs_ = graph.valid_inputs_;
    child_graph_order_ = graph.child_graph_order_;
    device_loop_ctrl_tensors_ = graph.device_loop_ctrl_tensors_;
    device_loop_ctrl_params_ = graph.device_loop_ctrl_params_;
    parent_graph_ = graph.parent_graph_;
    start_label_ = graph.start_label_;
    end_goto_ = graph.end_goto_;
    internal_parameter_to_front_node_map_ = graph.internal_parameter_to_front_node_map_;
    graph_output_to_front_node_map_ = graph.graph_output_to_front_node_map_;
    front_node_to_graph_output_map_ = graph.front_node_to_graph_output_map_;
    front_to_internal_outputs_map_ = graph.front_to_internal_outputs_map_;
    internal_outputs_to_front_map_ = graph.internal_outputs_to_front_map_;
    internal_outputs_tensor_map_ = graph.internal_outputs_tensor_map_;
    current_epoch_ = graph.current_epoch_;
    tuple_parameter_to_make_tuple_map_ = graph.tuple_parameter_to_make_tuple_map_;
    input_nodes_ = graph.input_nodes_;
    pre_graphs_ = graph.pre_graphs_;
    post_graphs_ = graph.post_graphs_;
    send_recv_pairs_for_parallel_op_inputs_ = graph.send_recv_pairs_for_parallel_op_inputs_;
    send_recv_pairs_for_parallel_op_outputs_ = graph.send_recv_pairs_for_parallel_op_outputs_;
    size_t pre_graph_finished_count = graph.pre_graph_finished_count_;
    pre_graph_finished_count_ = pre_graph_finished_count;
    size_t post_graph_finished_count = graph.post_graph_finished_count_;
    post_graph_finished_count_ = post_graph_finished_count;
    first_step_ = graph.first_step_;
    has_optimizer_ = graph.has_optimizer_;
    is_dynamic_shape_ = graph.is_dynamic_shape_;
    front_outputs_ = graph.front_outputs_;
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
  CNodePtr NewCNode(std::vector<AnfNodePtr> &&inputs) override;
  CNodePtr NewCNode(const std::vector<AnfNodePtr> &inputs = std::vector<AnfNodePtr>()) override;
  CNodePtr NewCNodeWithInfos(const std::vector<AnfNodePtr> &inputs, const CNodePtr &ori_cnode = nullptr);
  void CreateKernelInfoFromNewParameter(const CNodePtr &cnode) const;
  CNodePtr NewCNode(const CNodePtr &cnode);
  void ResetAssignInputFeatureMapFlag(const CNodePtr &cnode) const;
  ParameterPtr NewParameter(const ParameterPtr &parameter = nullptr);
  ParameterPtr NewParameter(const abstract::AbstractBasePtr &abstract);
  ValueNodePtr NewValueNode(const AbstractBasePtr &abstract, const ValuePtr &value) const;
  ValueNodePtr NewValueNode(const ValueNodePtr &value_node = nullptr) const;
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
  void SetNodeOutputEdges();
  uint32_t graph_id() const { return graph_id_; }
  void set_graph_id(uint32_t graph_id) { graph_id_ = graph_id; }
  uint32_t root_graph_id() const { return root_graph_id_; }
  void set_root_graph_id(uint32_t root_graph_id) { root_graph_id_ = root_graph_id; }
  DeviceType device_target() const { return device_target_; }
  void set_device_target(DeviceType target) { device_target_ = target; }

  // and a new front to backend anf relation to maop
  void FrontBackendMapAdd(const AnfNodePtr &front_anf, const AnfNodePtr &backend_anf);
  // replace old backend anf with new backend anf
  void FrontBackendlMapUpdate(const AnfNodePtr &old_backend_anf, const AnfNodePtr &new_backend_anf);
  // get backend anf by front anf
  AnfNodePtr GetBackendAnfByFrontAnf(const AnfNodePtr &front_anf);
  // get front anf by backend anf
  AnfNodePtr GetFrontAnfByBackendAnf(const AnfNodePtr &backend_anf) const;
  const mindspore::HashMap<AnfNodePtr, AnfNodePtr> &backend_front_anf_map() const { return backend_front_anf_map_; }
  // check backend node whether exist in map
  bool BackendNodeExistInFrontBackendMap(const AnfNodePtr &backend_anf);
  // get value node by tensor
  ValueNodePtr GetValueNodeByTensor(const tensor::TensorPtr &tensor);
  // add value node tensor relation map
  void TensorValueNodeMapAdd(const tensor::TensorPtr &tensor, const ValueNodePtr &value_node);
  // get all value nodes of graph
  const mindspore::HashSet<ValueNodePtr> graph_value_nodes() const { return graph_value_nodes_; }
  // add value node to graph
  void AddValueNodeToGraph(const ValueNodePtr &value_node);
  // ref output is in map
  bool IsInRefOutputMap(const AnfWithOutIndex &pair) const;
  // Whether the value corresponds to ref output.
  bool IsRefOutputMapValue(const AnfWithOutIndex &pair) const;
  // get ref correspond pairs
  AnfWithOutIndex GetRefCorrespondOutput(const AnfWithOutIndex &out_pair) const;
  // add ref correspond pairs
  void AddRefCorrespondPairs(const AnfWithOutIndex &final_pair, const AnfWithOutIndex &origin_pair);
  // Replace ref pair
  void ReplaceRefPair(const AnfWithOutIndex &old_pair, const AnfWithOutIndex &new_pair);
  // get map
  std::map<AnfWithOutIndex, AnfWithOutIndex> GetRefMap() const { return ref_out_in_map_; }
  // update ref map
  void set_ref_out_in_map(const std::map<AnfWithOutIndex, AnfWithOutIndex> &ref_out_in_map) {
    ref_out_in_map_ = ref_out_in_map;
  }
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
  // set need inline
  void set_need_inline(bool need_inline) { need_inline_ = need_inline; }
  // check whether need inline
  bool need_inline() const { return need_inline_; }
  // set invalid inputs for control sink
  std::vector<bool> *MutableValidInputs() { return &valid_inputs_; }
  std::vector<bool> valid_inputs() const { return valid_inputs_; }
  // replace node in graph
  void ReplaceNode(const AnfNodePtr &old_anf_node, const AnfNodePtr &new_anf_node);
  // set stream label of graph
  void set_stream_distinction_label(uint32_t stream_label) { stream_distinction_label_ = stream_label; }
  // get stream label of graph
  uint32_t stream_distinction_label() const { return stream_distinction_label_; }
  // refresh execute kernel stream label
  void UpdateExecuteKernelStreamLabel();
  // calculate the leaf graph order of root graph
  std::vector<std::shared_ptr<KernelGraph>> GetLeafGraphOrder();
  // the child graph of current graph
  const std::vector<std::weak_ptr<KernelGraph>> &child_graph_order() const { return child_graph_order_; }
  void set_child_graph_order(const std::vector<std::weak_ptr<KernelGraph>> &order) { child_graph_order_ = order; }
  // checkout whether current graph is leaf graph
  bool IsLeafGraph() const;

  void set_device_loop_ctrl_tensors(const std::map<std::string, tensor::TensorPtr> &device_loop_ctrl_tensors) {
    device_loop_ctrl_tensors_ = device_loop_ctrl_tensors;
  }
  std::map<std::string, tensor::TensorPtr> device_loop_control_tensors() const { return device_loop_ctrl_tensors_; }

  void set_device_loop_ctrl_params(const std::map<std::string, mindspore::ParameterPtr> &device_loop_ctrl_params) {
    device_loop_ctrl_params_ = device_loop_ctrl_params;
  }
  const std::map<std::string, mindspore::ParameterPtr> device_loop_control_params() const {
    return device_loop_ctrl_params_;
  }

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
  AnfWithOutIndex GetInternalOutputByFrontNode(const AnfNodePtr &front_node) const;
  bool IsInternalOutput(const AnfNodePtr &node, size_t output_idx) const;
  bool IsInternalOutput(const AnfNodePtr &node) const;
  bool IsUniqueTargetInternalOutput(const AnfNodePtr &node, size_t output_idx) const;
  void AddInternalOutputTensor(const AnfNodePtr &node, size_t output_idx, const tensor::TensorPtr &tensor);
  tensor::TensorPtr GetInternalOutputTensor(const AnfNodePtr &node, size_t output_idx);
  AnfWithOutIndex GetGraphOutputByFrontNode(const AnfWithOutIndex &front_node) const;

  // Cache the internal parameter and corresponding to front node into internal_parameter_to_front_node_map_.
  void CacheInternalParameterToFrontNode(const AnfNodePtr &parameter, const AnfWithOutIndex &front_node_with_index);
  // This function gets the real node that skip the monad control node.
  AnfWithOutIndex GetFrontNodeByInternalParameter(const AnfNodePtr &parameter) const;
  // This function gets the origin node used to connect monad controls between subgraphs.
  AnfWithOutIndex GetOriginFrontNodeByInternalParameter(const AnfNodePtr &parameter) const;

  // Get the funcgraph to which the kernel graph belongs.
  FuncGraphPtr GetFuncGraph();
  // Cache the backend graph output nodes and corresponding to front nodes with output index into
  // graph_output_to_front_node_map_.
  void CacheGraphOutputToFrontNodeWithIndex(const std::vector<AnfNodePtr> &backend_outputs,
                                            const std::vector<AnfNodePtr> &front_outputs);
  AnfWithOutIndex GetFrontNodeWithIndexByGraphOutput(const AnfWithOutIndex &backend_graph_output_with_index) const;

  void SetKernelObjectTypesForUnrealNodes();

  uint32_t current_epoch() const { return current_epoch_; }
  void set_current_epoch(uint32_t epoch) { current_epoch_ = epoch; }
  void UpdateChildGraphOrder();
  const std::vector<AnfNodePtr> &child_graph_result() const { return child_graph_result_; }
  void AddChildGraphResult(const AnfNodePtr &parameter) { child_graph_result_.push_back(parameter); }
  bool IsChildGraphResult(const AnfNodePtr &node);
  void set_child_graph_result(const std::vector<AnfNodePtr> &child_graph_result) {
    child_graph_result_ = child_graph_result;
  }

  void InsertTupleParameterToMakeTupleMap(const AnfNodePtr &param, const AnfNodePtr &make_tuple) {
    if (tuple_parameter_to_make_tuple_map_.find(param) != tuple_parameter_to_make_tuple_map_.end()) {
      return;
    }
    tuple_parameter_to_make_tuple_map_[param] = make_tuple;
  }
  AnfNodePtr FindTupleParameterToMakeTupleMap(const AnfNodePtr &param) const {
    if (tuple_parameter_to_make_tuple_map_.find(param) != tuple_parameter_to_make_tuple_map_.end()) {
      return tuple_parameter_to_make_tuple_map_.at(param);
    } else {
      return nullptr;
    }
  }
  void RemoveNodeFromGraph(const AnfNodePtr &node);
  void EnableRuntimeCache() const;
  void DisableRuntimeCache() const;
  void UpdateGraphDynamicAttr();
  void SetGraphDynamicAttr(bool is_dynamic_shape) { is_dynamic_shape_ = is_dynamic_shape; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  void UpdateGraphAquireGilAttr();
  void SetOptimizerFlag();
  void SetInputNodes();
  const std::vector<AnfNodePtr> &input_nodes() const { return input_nodes_; }
  void SetInputTensors(const std::vector<tensor::TensorPtr> &input_tensors) { input_tensors_ = input_tensors; }
  const std::vector<tensor::TensorPtr> &input_tensors() const { return input_tensors_; }

  void SetOutputNodeToTensor(const KernelMapTensor &node_to_tensor);

  tensor::TensorPtr GetNodeOutputTensor(const session::KernelWithIndex &output_index) const {
    auto iter = output_node_to_tensor_.find(output_index);
    if (iter != output_node_to_tensor_.end()) {
      return utils::cast<tensor::TensorPtr>(iter->second);
    }
    auto nop_node_output_iter = nop_node_output_map_.find(output_index);
    if (nop_node_output_iter != nop_node_output_map_.end()) {
      iter = output_node_to_tensor_.find(nop_node_output_iter->second);
      if (iter != output_node_to_tensor_.end()) {
        return utils::cast<tensor::TensorPtr>(iter->second);
      }
    }
    return nullptr;
  }

  bool has_optimizer() const { return has_optimizer_; }
  bool IsUpdatedParameter(const ParameterPtr &param) const {
    return updated_parameters_.find(param) != updated_parameters_.end();
  }
  // handle graph dependency
  void AddPreGraph(const std::shared_ptr<session::KernelGraph> &graph) {
    if (graph != nullptr) {
      pre_graphs_[graph->graph_id()] = graph;
    }
  }

  mindspore::HashMap<uint32_t, std::weak_ptr<session::KernelGraph>> get_pre_graphs() const { return pre_graphs_; }
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

  void IncPreGraphFinishedCount() { ++pre_graph_finished_count_; }
  void IncPostGraphFinishedCount() { ++post_graph_finished_count_; }
  void ResetGraphRunningStatus() {
    first_step_ = false;
    post_graph_finished_count_ = 0;
    pre_graph_finished_count_ = 0;
  }
  void OnRunGraphFinished() const {
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

  // The interface of parallel op send/recv pairs map.
  void InsertSendRecvPairForParallelOpInputs(const CNodePtr &parallel_op,
                                             const std::pair<CNodePtr, CNodePtr> &send_recv_pair) {
    auto iter = send_recv_pairs_for_parallel_op_inputs_.find(parallel_op);
    if (iter == send_recv_pairs_for_parallel_op_inputs_.end()) {
      send_recv_pairs_for_parallel_op_inputs_[parallel_op] = {send_recv_pair};
    } else {
      iter->second.emplace_back(send_recv_pair);
    }
  }

  void InsertSendRecvPairForParallelOpOutputs(const CNodePtr &parallel_op,
                                              const std::pair<CNodePtr, CNodePtr> &send_recv_pair) {
    auto iter = send_recv_pairs_for_parallel_op_outputs_.find(parallel_op);
    if (iter == send_recv_pairs_for_parallel_op_outputs_.end()) {
      send_recv_pairs_for_parallel_op_outputs_[parallel_op] = {send_recv_pair};
    } else {
      iter->second.emplace_back(send_recv_pair);
    }
  }

  const mindspore::HashMap<CNodePtr, std::vector<std::pair<CNodePtr, CNodePtr>>>
    &send_recv_pairs_for_parallel_op_inputs() const {
    return send_recv_pairs_for_parallel_op_inputs_;
  }
  const mindspore::HashMap<CNodePtr, std::vector<std::pair<CNodePtr, CNodePtr>>>
    &send_recv_pairs_for_parallel_op_outputs() const {
    return send_recv_pairs_for_parallel_op_outputs_;
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

  std::map<AnfWithOutIndex, AnfWithOutIndex> graph_output_map() { return graph_output_to_front_node_map_; }
  std::map<AnfWithOutIndex, AnfWithOutIndex> front_node_to_graph_output_map() {
    return front_node_to_graph_output_map_;
  }

  // The interface to set/get the graph GIL flag.
  void set_is_need_gil(bool flag) { is_need_gil_ = flag; }
  bool is_need_gil() const { return is_need_gil_; }

  bool IsDatasetGraph() const;

  void set_is_from_single_op(bool is_from_single_op) { is_from_single_op_ = is_from_single_op; }
  bool is_from_single_op() const { return is_from_single_op_; }
  void set_run_mode(device::RunMode run_mode) { run_mode_ = run_mode; }
  bool is_graph_run_mode() const { return run_mode_ == device::RunMode::kGraphMode; }
  bool is_loop_count_sink() const { return is_loop_count_sink_; }
  void set_is_loop_count_sink(bool is_loop_count_sink) { is_loop_count_sink_ = is_loop_count_sink; }
  const mindspore::HashMap<AnfNodePtr, AnfNodePtr> &front_backend_anf_map() const { return front_backend_anf_map_; }

  AnfWithOutIndex GetElementInTupleBackendFrontIndexMap(const AnfNodePtr &back_node) const {
    auto iter = tuple_backend_front_anf_index_map_.find(back_node);
    if (iter == tuple_backend_front_anf_index_map_.end()) {
      return AnfWithOutIndex(nullptr, 0);
    }
    return iter->second;
  }

  AnfNodePtrList front_outputs() const { return front_outputs_; }
  void set_front_outputs(const AnfNodePtrList &outputs) { front_outputs_ = outputs; }
  bool IsCommSubGraph(uint32_t id) const { return comm_sub_graph_ids_.find(id) != comm_sub_graph_ids_.end(); }
  void RecordNewCommSubGraphId(uint32_t id) { comm_sub_graph_ids_.insert(id); }

  // somas total memory size
  SomasInfo *MutableSomasInfo() const { return somas_info_.get(); }
  size_t somas_whole_block_size() const { return somas_info_->whole_block_size_; }
  const std::map<size_t, size_t> &somas_merged_blocks_map() const { return somas_info_->merged_blocks_map_; }

 private:
  // remove value node form graph
  bool RemoveValueNodeFromGraph(const ValueNodePtr &value_node);
  void SetKernelInfoForNode(const AnfNodePtr &node) const;
  AnfNodePtr MakeValueNode(const AnfNodePtr &node) const;

  AnfNodePtr TransValueNodeTuple(const AbstractBasePtr &abstract, const ValuePtr &value);
  AnfNodePtr TransParameterTuple(const AbstractBasePtr &abstract);
  AnfNodePtr TransCNodeTuple(const CNodePtr &node);
  AnfNodePtr CreatTupleGetItemNode(const AnfNodePtr &node, size_t output_idx);
  std::vector<CNodePtr> SortStartLabelAndEndGoto();
  void PostNewCNode(const CNodePtr &cnode) const;

  // members
  std::shared_ptr<std::vector<AnfNodePtr>> inputs_;
  std::shared_ptr<SomasInfo> somas_info_;
  std::vector<AnfNodePtr> child_graph_result_;
  std::vector<CNodePtr> execution_order_;
  std::vector<CNodePtr> mem_reuse_exec_order_;
  uint32_t graph_id_;
  uint32_t stream_distinction_label_;
  DeviceType device_target_;
  uint32_t root_graph_id_{0};

  // record map bettween front anf and backend anf,use two map implement bidirectional map
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> front_backend_anf_map_;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> backend_front_anf_map_;
  mindspore::HashMap<AnfNodePtr, AnfWithOutIndex> tuple_backend_front_anf_index_map_;
  // there may be a tensor from ME backend ,a value ndoe will be create according the tensor,map record
  mindspore::HashMap<tensor::TensorPtr, ValueNodePtr> tensor_to_value_node_map_;
  // include all value nodes
  mindspore::HashSet<ValueNodePtr> graph_value_nodes_;
  // record map between ref final output anf with index and ref origin input with index
  std::map<AnfWithOutIndex, AnfWithOutIndex> ref_out_in_map_;
  mindspore::HashMap<AnfNodePtr, std::vector<AnfNodePtr>> node_output_edges_;
  std::map<std::string, std::pair<AnfNodePtr, int>> summary_nodes_;
  // parameters that will be updated when graph is executed
  mindspore::HashSet<ParameterPtr> updated_parameters_;

  // graph needn't execute
  bool executable_{false};
  // exist summary node in graph
  bool summary_node_exist_{false};
  // valid inputs
  std::vector<bool> valid_inputs_;
  // need inline
  bool need_inline_;

  // child graph execute order in parent graph
  std::vector<std::weak_ptr<KernelGraph>> child_graph_order_;

  // device loop control frontend tensors
  std::map<std::string, tensor::TensorPtr> device_loop_ctrl_tensors_;
  // device loop control backend nodes
  std::map<std::string, mindspore::ParameterPtr> device_loop_ctrl_params_;

  // parameter graph
  std::weak_ptr<KernelGraph> parent_graph_;

  CNodePtr start_label_;
  CNodePtr end_goto_;

  AnfNodePtrList front_outputs_;
  // Internal parameter is not the origin parameter of func graph, it is the output of previous kernel graph which is
  // related to the input of this kernel graph. The first of unordered map is the input of this kernel graph, the second
  // of unordered map is front node corresponding to the output of previous kernel graph.
  mindspore::HashMap<AnfNodePtr, AnfWithOutIndex> internal_parameter_to_front_node_map_;
  // The first of map is the backend graph output of this kernel graph, the second of map is front node corresponding to
  // the backend node with index.
  std::map<AnfWithOutIndex, AnfWithOutIndex> graph_output_to_front_node_map_;
  std::map<AnfWithOutIndex, AnfWithOutIndex> front_node_to_graph_output_map_;

  mindspore::HashMap<AnfNodePtr, AnfWithOutIndex> front_to_internal_outputs_map_;
  mindspore::HashMap<AnfNodePtr, mindspore::HashMap<size_t, std::pair<AnfNodePtr, bool>>>
    internal_outputs_to_front_map_;
  mindspore::HashMap<AnfNodePtr, mindspore::HashMap<size_t, tensor::TensorPtr>> internal_outputs_tensor_map_;
  uint32_t current_epoch_;
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> tuple_parameter_to_make_tuple_map_;
  std::vector<AnfNodePtr> input_nodes_;
  std::vector<tensor::TensorPtr> input_tensors_;
  KernelMapTensor output_node_to_tensor_;
  std::map<session::KernelWithIndex, session::KernelWithIndex, session::KernelWithIndexCmp> nop_node_output_map_;
  mindspore::HashMap<uint32_t, std::weak_ptr<session::KernelGraph>> pre_graphs_;
  mindspore::HashMap<uint32_t, std::weak_ptr<session::KernelGraph>> post_graphs_;

  // key:parallel op ptr, value:vector of <send op receive op > pairs
  mindspore::HashMap<CNodePtr, std::vector<std::pair<CNodePtr, CNodePtr>>> send_recv_pairs_for_parallel_op_inputs_;
  // key:parallel op ptr, value:vector of <send op receive op > pairs
  mindspore::HashMap<CNodePtr, std::vector<std::pair<CNodePtr, CNodePtr>>> send_recv_pairs_for_parallel_op_outputs_;
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

  // Indicate whether the kernels in the graphs acquire Python GIL.
  bool is_need_gil_{false};

  // Indicate whether the kernel graph is constructed from single op in function graph
  bool is_from_single_op_{false};

  // Indicate whether the kernel graph sink will run on graph executor or kernel executor
  device::RunMode run_mode_{device::RunMode::kUnknown};

  // Indicate whether the kernel graph loop sink to the device executing.
  bool is_loop_count_sink_{false};
  // save the communication sub-graph id for comm op reuse
  std::set<uint32_t> comm_sub_graph_ids_{};
};
}  // namespace session
using KernelGraphPtr = std::shared_ptr<session::KernelGraph>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_KERNEL_GRAPH_H
