/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OP_TRACER_H_
#define MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OP_TRACER_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "device_address/device_address.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// Entry for address-based fallback lookup with weak_ptr validation.
struct AddressNodeEntry {
  AnfNodePtr node;
  std::weak_ptr<device::DeviceAddress> weak_addr;
};

class OpTracer {
 public:
  OpTracer() : func_graph_(std::make_shared<FuncGraph>()) {
    func_graph_->debug_info()->set_name("hyper_offload_traced_graph");
  }
  ~OpTracer() = default;

  // Trace a single operator execution.
  // Called by HyperOffloadContext during TRACING state.
  void TraceOp(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output);

  // Finalize the graph by setting the output (return node).
  // Finds root nodes (CNodes not used by other CNodes) and creates MakeTuple as output.
  bool FinalizeGraph();

  // Get the traced FuncGraph.
  FuncGraphPtr GetTracedGraph() const { return func_graph_; }

  // Get the execution order of CNodes.
  const std::vector<CNodePtr> &GetExecutionOrder() const { return execution_order_; }

  // Clear all traced data and reset the graph.
  void Reset();

 private:
  // Get or create an AnfNode for the given value.
  // Uses tensor_id as primary key, device_address as fallback key.
  // This handles view ops (different tensor_id) and inplace ops (same tensor_id).
  // Also handles tensor copy case where tensor_id changes but address remains same.
  AnfNodePtr GetOrCreateNode(const ValuePtr &value);

  // Get node by tensor, trying tensor_id first, then address fallback.
  AnfNodePtr GetNodeByTensor(const tensor::TensorPtr &tensor);

  // Register output node with tensor_id as key.
  void RegisterOutputNode(const ValuePtr &value, const AnfNodePtr &node);

  // Recursively register output nodes with TupleGetItem chain for nested sequences.
  void RegisterOutputNode(const ValuePtr &value, const AnfNodePtr &node, const std::vector<int64_t> &index_path);

  // Create a parameter node for input tensor.
  ParameterPtr CreateParameterNode(const tensor::TensorPtr &tensor);

  // Create a value node for non-tensor values.
  ValueNodePtr CreateValueNode(const ValuePtr &value);

  // The traced FuncGraph.
  FuncGraphPtr func_graph_;

  // Primary key: tensor_id -> node.
  // tensor_id is globally unique and never reused.
  std::unordered_map<uint64_t, AnfNodePtr> tensor_id_to_node_;

  // Fallback key: device_address -> (node, weak_ptr).
  // Used when tensor_id changes but address remains same (e.g., tensor copy).
  // weak_ptr is used to validate that the address hasn't been reused by a different tensor.
  std::unordered_map<const void *, AddressNodeEntry> addr_to_node_;

  // Map from parameter index to parameter node for graph inputs.
  std::vector<ParameterPtr> input_params_;

  // Execution order of CNodes (actual operators).
  std::vector<CNodePtr> execution_order_;

  // Virtual nodes created during tracing (MakeTuple, TupleGetItem).
  // These are not actual operators but are needed for graph structure.
  // Used in FinalizeGraph to correctly identify root nodes.
  std::vector<CNodePtr> virtual_nodes_;
};

using OpTracerPtr = std::shared_ptr<OpTracer>;

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PYNATIVE_HYPER_OFFLOAD_OP_TRACER_H_
