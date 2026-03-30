/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "pynative/hyper_offload/op_tracer.h"
#include <algorithm>
#include <mutex>
#include <unordered_set>
#include "utils/log_adapter.h"
#include "utils/info.h"
#include "utils/trace_info.h"
#include "include/utils/utils.h"
#include "include/pynative/utils/pynative_utils.h"
#include "ir/core_ops_primitive.h"
#include "utils/ir_dump/anf_ir_dump.h"

namespace mindspore {
namespace pynative {
namespace hyper_offload {

// Note:  registration is now handled by HyperOffloadContext at program startup.
// OpTracer is PostHooknow a pure graph tracer, called by HyperOffloadContext.

namespace {
// Capture the current Python call stack and set it as debug info on the CNode.
// This enables DumpIR to show source code locations for each traced operator.
// GetPythonStack() returns frames ordered outermost (index 0) to innermost (last).
void SetPythonCallStackDebugInfo(const CNodePtr &cnode) {
  const auto &stacks = GetPythonStack();
  if (stacks.empty()) {
    return;
  }

  // Set location on the CNode's existing debug_info using the innermost frame.
  const auto &innermost = stacks.back();
  auto innermost_loc = std::make_shared<Location>(std::get<0>(innermost), std::get<1>(innermost), 0,
                                                  std::get<1>(innermost), 0, "", std::vector<std::string>{});
  cnode->debug_info()->set_location(innermost_loc);

  // Chain remaining frames from innermost toward outermost via trace_info links,
  // so DumpIR's CollectDebugInfos() walks the chain and outputs the full stack
  // in the standard order: [outermost, ..., innermost].
  DebugInfoPtr prev_info = cnode->debug_info();
  for (int i = static_cast<int>(stacks.size()) - 2; i >= 0; --i) {
    const auto &frame = stacks[i];
    auto loc = std::make_shared<Location>(std::get<0>(frame), std::get<1>(frame), 0, std::get<1>(frame), 0, "",
                                          std::vector<std::string>{});
    auto parent_info = std::make_shared<DebugInfo>(loc);
    prev_info->set_trace_info(std::make_shared<TraceInfo>(parent_info));
    prev_info = parent_info;
  }
}

// Helper: Check if any input tensor shares device_address with given output tensor.
bool IsInplaceOp(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  return prim != nullptr && prim->name().find("Inplace") != std::string::npos;
}

// Mark view/inplace operators by checking if input and output share device_address.
// Supports single-output and multi-output (ValueTuple) cases.
void MarkViewInplaceOp(const CNodePtr &cnode, const std::vector<ValuePtr> &inputs, const ValuePtr &output) {
  if (output == nullptr || inputs.empty()) {
    return;
  }

  bool is_view = false;
  bool is_inplace = false;
  // Case 1: Single tensor output
  if (output->isa<tensor::Tensor>()) {
    auto output_tensor = output->cast<tensor::TensorPtr>();
    if (IsInplaceOp(cnode)) {
      is_inplace = true;
    }
    if (output_tensor->storage_info() != nullptr) {
      is_view = true;
    }
  } else if (output->isa<ValueSequence>()) {  // Case 2: Multi-output (ValueTuple, e.g., split)
    auto seq = output->cast<ValueSequencePtr>();
    for (const auto &elem : seq->value()) {
      if (elem != nullptr && elem->isa<tensor::Tensor>()) {
        auto output_tensor = elem->cast<tensor::TensorPtr>();
        if (IsInplaceOp(cnode)) {
          is_inplace = true;
        }
        if (output_tensor->storage_info() != nullptr) {
          is_view = true;
        }
      }
    }
  }

  if (is_view) {
    cnode->set_user_data("is_view", std::make_shared<bool>(true));
    MS_LOG(DEBUG) << "Marked CNode as view: " << cnode->DebugString();
  }
  if (is_inplace) {
    cnode->set_user_data("is_inplace", std::make_shared<bool>(true));
    MS_LOG(DEBUG) << "Marked CNode as inplace: " << cnode->DebugString();
  }
}
}  // namespace

void OpTracer::Reset() {
  func_graph_ = std::make_shared<FuncGraph>();
  func_graph_->debug_info()->set_name("hyper_offload_traced_graph");
  tensor_id_to_node_.clear();
  addr_to_node_.clear();
  input_params_.clear();
  execution_order_.clear();
  virtual_nodes_.clear();
}

void OpTracer::TraceOp(const std::string &op_name, const std::vector<ValuePtr> &inputs, const ValuePtr &output) {
  MS_LOG(DEBUG) << "TraceOp: " << op_name << ", inputs size: " << inputs.size();

  // Build CNode inputs: [primitive, input1, input2, ...]
  AnfNodePtrList cnode_inputs;

  // Create primitive node.
  auto prim = std::make_shared<Primitive>(op_name);
  auto prim_node = NewValueNode(prim);
  (void)cnode_inputs.emplace_back(prim_node);

  // Get or create nodes for each input.
  for (const auto &input : inputs) {
    auto node = GetOrCreateNode(input);
    if (node != nullptr) {
      (void)cnode_inputs.emplace_back(node);
    }
  }

  // Create CNode for this operation.
  auto cnode = func_graph_->NewCNodeInOrder(cnode_inputs);
  MS_LOG(DEBUG) << "Created CNode for " << op_name << ": " << cnode->DebugString();

  // Capture Python call stack only in sync mode: in async mode the GIL may not be held
  // at the point of actual execution, and the stack would point to the wrong place.
  if (!PyNativeAlgo::Common::GetPyNativeExecutor()->forward_executor()->enable_async()) {
    SetPythonCallStackDebugInfo(cnode);
  }

  // Set abstract for the CNode.
  if (output != nullptr) {
    cnode->set_abstract(output->ToAbstract()->Broaden());
  }

  // Add to execution order for finding root nodes later.
  (void)execution_order_.emplace_back(cnode);

  // Mark view/inplace operators by checking if input and output share device_address.
  MarkViewInplaceOp(cnode, inputs, output);
  // Register output node with device address mapping.
  RegisterOutputNode(output, cnode);
}

AnfNodePtr OpTracer::GetOrCreateNode(const ValuePtr &value) {
  if (value == nullptr) {
    return nullptr;
  }

  // Handle Tensor type - use tensor_id as primary key, address as fallback.
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();

    // Try to find existing node (tensor_id first, then address fallback).
    auto node = GetNodeByTensor(tensor);
    if (node != nullptr) {
      MS_LOG(DEBUG) << "Found existing node for tensor_id: " << tensor->id();
      return node;
    }

    // Not found, create a new node for this input tensor.
    // Use is_parameter() to determine if it's a weight (Parameter) or input data (ValueNode).
    AnfNodePtr new_node;
    if (tensor->is_parameter()) {
      new_node = CreateParameterNode(tensor);
    } else {
      // Input data tensor: create fake tensor ValueNode.
      auto copy_tensor = std::make_shared<tensor::Tensor>(*tensor);
      copy_tensor->set_device_address(nullptr);
      new_node = CreateValueNode(copy_tensor);
    }

    // Register with both tensor_id (primary) and address (fallback).
    uint64_t tensor_id = tensor->id();
    tensor_id_to_node_[tensor_id] = new_node;

    const auto &device_addr = tensor->device_address();
    if (device_addr != nullptr) {
      AddressNodeEntry entry;
      entry.node = new_node;
      entry.weak_addr = device_addr;
      addr_to_node_[device_addr.get()] = entry;
      MS_LOG(DEBUG) << "Registered new node for tensor_id=" << tensor_id << ", addr=" << device_addr.get();
    } else {
      MS_LOG(DEBUG) << "Registered new node for tensor_id=" << tensor_id << " (no device_address)";
    }

    return new_node;
  }

  // Handle ValueSequence (tuple/list of values) - recursively process.
  if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    const auto &v_list = seq->value();

    // Check if this is a pure non-tensor sequence (e.g., tuple of scalars).
    bool has_tensor = std::any_of(v_list.begin(), v_list.end(), [](const ValuePtr &elem) {
      if (elem->isa<tensor::Tensor>()) {
        return true;
      }
      if (elem->isa<ValueSequence>()) {
        // Recursively check nested sequences.
        return true;  // Assume nested sequences may contain tensors.
      }
      return false;
    });

    if (!has_tensor) {
      // Pure non-tensor sequence, create ValueNode directly.
      return CreateValueNode(value);
    }

    // Build MakeTuple node by recursively getting nodes for each element.
    AnfNodePtrList make_tuple_inputs;
    abstract::AbstractBasePtrList abs_list;
    (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));

    for (const auto &elem : v_list) {
      auto elem_node = GetOrCreateNode(elem);
      if (elem_node != nullptr) {
        (void)make_tuple_inputs.emplace_back(elem_node);
        (void)abs_list.emplace_back(elem->ToAbstract()->Broaden());
      }
    }
    auto make_tuple_node = func_graph_->NewCNodeInOrder(make_tuple_inputs);
    make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    // Add to virtual_nodes_ for correct root node calculation in FinalizeGraph.
    (void)virtual_nodes_.emplace_back(make_tuple_node);
    MS_LOG(DEBUG) << "Created MakeTuple node: " << make_tuple_node->DebugString();
    return make_tuple_node;
  }

  // For other value types (scalars, None, etc.), create a ValueNode.
  return CreateValueNode(value);
}

AnfNodePtr OpTracer::GetNodeByTensor(const tensor::TensorPtr &tensor) {
  if (tensor == nullptr) {
    return nullptr;
  }

  // Step 1: Try primary key (tensor_id).
  uint64_t tensor_id = tensor->id();
  auto id_it = tensor_id_to_node_.find(tensor_id);
  if (id_it != tensor_id_to_node_.end()) {
    MS_LOG(DEBUG) << "Found node by tensor_id=" << tensor_id;
    return id_it->second;
  }

  // Step 2: Fallback to address lookup with weak_ptr validation.
  const auto &device_addr = tensor->device_address();
  if (device_addr == nullptr) {
    return nullptr;
  }

  const void *addr_key = device_addr.get();
  auto addr_it = addr_to_node_.find(addr_key);
  if (addr_it == addr_to_node_.end()) {
    return nullptr;
  }

  // Validate with weak_ptr: check if the original DeviceAddress is still alive.
  auto &entry = addr_it->second;
  auto locked_addr = entry.weak_addr.lock();
  if (locked_addr == nullptr) {
    // Original address has been released, this is a reused address.
    // Remove stale entry and return nullptr.
    MS_LOG(DEBUG) << "Address " << addr_key << " was reused (weak_ptr expired), removing stale entry.";
    addr_to_node_.erase(addr_it);
    return nullptr;
  }

  MS_LOG(DEBUG) << "Found node by address fallback, addr=" << addr_key;
  return entry.node;
}

void OpTracer::RegisterOutputNode(const ValuePtr &value, const AnfNodePtr &node) {
  RegisterOutputNode(value, node, {});
}

void OpTracer::RegisterOutputNode(const ValuePtr &value, const AnfNodePtr &node,
                                  const std::vector<int64_t> &index_path) {
  if (value == nullptr || node == nullptr) {
    return;
  }

  // Handle single Tensor output.
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    uint64_t tensor_id = tensor->id();

    // Build TupleGetItem chain if we have index path.
    AnfNodePtr current_node = node;
    for (int64_t idx : index_path) {
      AnfNodePtrList tuple_getitem_inputs;
      (void)tuple_getitem_inputs.emplace_back(NewValueNode(prim::kPrimTupleGetItem));
      (void)tuple_getitem_inputs.emplace_back(current_node);
      (void)tuple_getitem_inputs.emplace_back(NewValueNode(idx));
      auto getitem_node = func_graph_->NewCNodeInOrder(tuple_getitem_inputs);

      // Get abstract for this level.
      auto current_abs = current_node->abstract();
      if (current_abs != nullptr && current_abs->isa<abstract::AbstractSequence>()) {
        auto seq_abs = current_abs->cast<abstract::AbstractSequencePtr>();
        if (idx < static_cast<int64_t>(seq_abs->elements().size())) {
          getitem_node->set_abstract(seq_abs->elements()[idx]);
        }
      }

      // Add to virtual_nodes_ for correct root node calculation in FinalizeGraph.
      (void)virtual_nodes_.emplace_back(getitem_node);

      current_node = getitem_node;
    }

    // Register with both tensor_id (primary) and address (fallback).
    // For inplace ops: same tensor_id will overwrite the previous node (correct behavior).
    // For view ops: different tensor_id will create a new entry (different node).
    tensor_id_to_node_[tensor_id] = current_node;

    const auto &device_addr = tensor->device_address();
    if (device_addr != nullptr) {
      AddressNodeEntry entry;
      entry.node = current_node;
      entry.weak_addr = device_addr;
      addr_to_node_[device_addr.get()] = entry;
      MS_LOG(DEBUG) << "Registered output node for tensor_id=" << tensor_id << ", addr=" << device_addr.get()
                    << ", index_path size: " << index_path.size();
    } else {
      MS_LOG(DEBUG) << "Registered output node for tensor_id=" << tensor_id
                    << ", index_path size: " << index_path.size() << " (no device_address)";
    }
    return;
  }

  // Handle ValueSequence (multiple outputs) - recursively process.
  if (value->isa<ValueSequence>()) {
    auto seq = value->cast<ValueSequencePtr>();
    for (size_t i = 0; i < seq->size(); ++i) {
      const auto &elem = seq->value()[i];
      // Build new index path for recursive call.
      std::vector<int64_t> new_index_path = index_path;
      new_index_path.emplace_back(static_cast<int64_t>(i));
      // Recursively register each element.
      RegisterOutputNode(elem, node, new_index_path);
    }
  }
}

ParameterPtr OpTracer::CreateParameterNode(const tensor::TensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = func_graph_->add_parameter();
  param->set_abstract(tensor->ToAbstract()->Broaden());
  input_params_.emplace_back(param);
  MS_LOG(DEBUG) << "Created parameter node for tensor, shape: " << tensor->shape();
  return param;
}

ValueNodePtr OpTracer::CreateValueNode(const ValuePtr &value) {
  auto value_node = NewValueNode(value);
  value_node->set_abstract(value->ToAbstract()->Broaden());
  return value_node;
}

bool OpTracer::FinalizeGraph() {
  if (execution_order_.empty()) {
    MS_LOG(WARNING) << "No operators traced, cannot finalize graph.";
    return false;
  }

  // Build a set of all nodes that are used as inputs by other CNodes.
  // Include both execution_order_ (actual ops) and virtual_nodes_ (MakeTuple, TupleGetItem).
  std::unordered_set<AnfNodePtr> used_nodes;

  // Collect inputs from actual operator CNodes.
  for (const auto &cnode : execution_order_) {
    // Skip the first input (primitive)
    for (size_t i = 1; i < cnode->size(); ++i) {
      (void)used_nodes.insert(cnode->input(i));
    }
  }

  // Collect inputs from virtual nodes (MakeTuple, TupleGetItem).
  // This ensures that nodes used by TupleGetItem are not incorrectly marked as root nodes.
  for (const auto &cnode : virtual_nodes_) {
    // Skip the first input (primitive)
    for (size_t i = 1; i < cnode->size(); ++i) {
      (void)used_nodes.insert(cnode->input(i));
    }
  }

  // Find root nodes: CNodes in execution_order_ that are not used by any other CNode.
  // Note: We only consider execution_order_ for root nodes, not virtual_nodes_.
  std::vector<AnfNodePtr> root_nodes;
  for (const auto &cnode : execution_order_) {
    if (used_nodes.find(cnode) == used_nodes.end()) {
      (void)root_nodes.emplace_back(cnode);
    }
  }

  MS_LOG(DEBUG) << "Found " << root_nodes.size() << " root nodes out of " << execution_order_.size() << " CNodes (with "
                << virtual_nodes_.size() << " virtual nodes).";

  if (root_nodes.empty()) {
    MS_LOG(WARNING) << "No root nodes found, using last CNode as output.";
    (void)root_nodes.emplace_back(execution_order_.back());
  }

  // Build output node.
  AnfNodePtr output_node;
  if (root_nodes.size() == 1) {
    // Single root node, use it directly as output.
    output_node = root_nodes[0];
  } else {
    // Multiple root nodes, create MakeTuple to combine them.
    AnfNodePtrList make_tuple_inputs;
    make_tuple_inputs.reserve(root_nodes.size() + 1);
    abstract::AbstractBasePtrList abs_list;
    abs_list.reserve(root_nodes.size());
    (void)make_tuple_inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));

    for (const auto &root : root_nodes) {
      make_tuple_inputs.emplace_back(root);
      if (root->abstract() != nullptr) {
        (void)abs_list.emplace_back(root->abstract());
      }
    }

    output_node = func_graph_->NewCNodeInOrder(make_tuple_inputs);
    output_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
    MS_LOG(DEBUG) << "Created MakeTuple for " << root_nodes.size() << " root nodes.";
  }

  // Set the output of the func_graph.
  func_graph_->set_output(output_node);
  MS_LOG(INFO) << "FuncGraph finalized with output: " << output_node->DebugString();
  return true;
}

}  // namespace hyper_offload
}  // namespace pynative
}  // namespace mindspore
