/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/pynative/ir_converter.h"

#include <atomic>
#include <algorithm>
#include <map>
#include <string>
#include <unordered_map>
#include "include/backend/anf_runtime_algorithm.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace pynative {
namespace {
constexpr auto kEdge = "Edge";
constexpr auto kSingleOp = "SingleOp";
constexpr auto kSimpleGraph = "SimpleGraph";
constexpr auto kName = "name";
constexpr auto kType = "type";
constexpr auto kOpName = "op_name";
constexpr auto kInputs = "inputs";
constexpr auto kOutputs = "outputs";
constexpr auto kEdgeTypeParameter = "ParameterEdge";
constexpr auto kEdgeTypeValueNode = "ValueNodeEdge";
constexpr auto kEdgeTypeOpOutput = "OpOutputEdge";

uint64_t MakeEdgeId() {
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}

uint64_t MakeOpId() {
  static std::atomic<uint64_t> last_id{1};
  return last_id.fetch_add(1, std::memory_order_relaxed);
}

std::string EdgeEnumToString(EdgeType edge_type) {
  static const std::unordered_map<EdgeType, std::string> edge_to_string = {
    {EdgeType::kParameterEdge, kEdgeTypeParameter},
    {EdgeType::kValueNodeEdge, kEdgeTypeValueNode},
    {EdgeType::kOpOutputEdge, kEdgeTypeOpOutput},
  };
  auto iter = edge_to_string.find(edge_type);
  if (iter == edge_to_string.end()) {
    MS_LOG(EXCEPTION) << "Unknown edge type " << edge_type;
  }
  return iter->second;
}

std::vector<session::KernelWithIndex> GetGraphOutputs(const KernelGraphPtr &graph) {
  const auto &output_nodes = graph->outputs();
  std::vector<session::KernelWithIndex> outputs;
  outputs.reserve(output_nodes.size());
  std::transform(output_nodes.begin(), output_nodes.end(), std::back_inserter(outputs),
                 [](const AnfNodePtr &node) { return common::AnfAlgo::VisitKernel(node, 0); });
  return outputs;
}

void ConvertValueNodes(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                       std::map<device::DeviceAddress *, EdgePtr> *address_to_edge) {
  const auto &value_nodes = graph->graph_value_nodes();
  for (const auto &value_node : value_nodes) {
    MS_EXCEPTION_IF_NULL(value_node);
    const auto &value = value_node->value();
    MS_EXCEPTION_IF_NULL(value);
    if (value->isa<Primitive>()) {
      // There are many Primitives in graph_value_nodes when running @trace.
      MS_LOG(WARNING) << "Skip converting CNode Primitive to Edge. ValueNode is " << value_node->DebugString();
      continue;
    }
    if (!AnfAlgo::OutputAddrExist(value_node, 0, false)) {
      MS_LOG(EXCEPTION) << "ValueNode " << value_node->DebugString() << " has no DeviceAddress";
    }
    auto node_address = AnfAlgo::GetMutableOutputAddr(value_node, 0, false);
    auto cloned_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
    auto edge =
      std::make_shared<Edge>(EdgeType::kValueNodeEdge, node_address, cloned_address, std::make_pair(value_node, 0));
    (*address_to_edge)[node_address.get()] = edge;
  }
}

std::vector<EdgePtr> ConvertGraphInputs(const KernelGraphPtr &graph, const device::DeviceContext *device_context,
                                        std::map<device::DeviceAddress *, EdgePtr> *address_to_edge) {
  const auto &inputs = graph->inputs();
  std::vector<EdgePtr> graph_inputs_edges;
  graph_inputs_edges.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    auto node_address = AnfAlgo::GetMutableOutputAddr(input, 0);
    MS_EXCEPTION_IF_NULL(node_address);

    auto cloned_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
    auto edge =
      std::make_shared<Edge>(EdgeType::kParameterEdge, node_address, cloned_address, std::make_pair(input, 0));
    (*address_to_edge)[node_address.get()] = edge;
    graph_inputs_edges.push_back(edge);
  }
  return graph_inputs_edges;
}

std::vector<EdgePtr> ConvertGraphOutputs(const KernelGraphPtr &graph,
                                         const std::map<device::DeviceAddress *, EdgePtr> &address_to_edge) {
  const auto &graph_outputs = GetGraphOutputs(graph);
  std::vector<EdgePtr> graph_outputs_edges;
  graph_outputs_edges.reserve(graph_outputs.size());
  for (const auto &[output, index] : graph_outputs) {
    auto output_address = AnfAlgo::GetMutableOutputAddr(output, index, false);
    auto iter = address_to_edge.find(output_address.get());
    if (iter == address_to_edge.end()) {
      MS_LOG(EXCEPTION) << "The node " << output->DebugString() << " index " << index << " have no edges.";
    }
    graph_outputs_edges.push_back(iter->second);
  }
  return graph_outputs_edges;
}

std::vector<EdgePtr> ConvertSingleOpInputEdges(const CNodePtr &node,
                                               const std::map<device::DeviceAddress *, EdgePtr> &address_to_edge) {
  // Get SingleOp inputs
  auto input_num = common::AnfAlgo::GetInputTensorNum(node);
  std::vector<EdgePtr> input_edges;
  input_edges.reserve(input_num);
  for (size_t i = 0; i < input_num; ++i) {
    const auto &device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(node, i, false);
    auto iter = address_to_edge.find(device_address.get());
    if (iter != address_to_edge.end()) {
      input_edges.push_back(iter->second);
    } else {
      // The input of this node is graph input or previous output, and it must had edge.
      MS_LOG(EXCEPTION) << "Cannot found input edge for " << node->DebugString() << " with input node "
                        << node->DebugString() << " input index " << i;
    }
  }
  return input_edges;
}

std::vector<EdgePtr> ConvertSingleOpOutputEdges(const CNodePtr &node, const device::DeviceContext *device_context,
                                                std::map<device::DeviceAddress *, EdgePtr> *address_to_edge) {
  auto output_num = AnfAlgo::GetOutputTensorNum(node);
  std::vector<EdgePtr> output_edges;
  output_edges.reserve(output_num);

  for (size_t i = 0; i < output_num; ++i) {
    auto node_address = AnfAlgo::GetMutableOutputAddr(node, i, false);
    // For ref node.
    auto iter = address_to_edge->find(node_address.get());
    if (iter != address_to_edge->end()) {
      const auto &edge = iter->second;
      output_edges.push_back(edge);
      continue;
    }

    auto clone_address = runtime::DeviceAddressUtils::CloneEmptyDeviceAddress(node_address, device_context);
    auto edge = std::make_shared<Edge>(EdgeType::kOpOutputEdge, node_address, clone_address, std::make_pair(node, i));
    output_edges.push_back(edge);
    (*address_to_edge)[node_address.get()] = edge;
  }
  return output_edges;
}
}  // namespace

SimpleGraphPtr IrConverter::Convert(const std::string &name, const KernelGraphPtr &graph,
                                    const device::DeviceContext *device_context) {
  // Same DeviceAddress, same EdgePtr. Such RefNode input/output DeviceAddress.
  std::map<device::DeviceAddress *, EdgePtr> address_to_edge;

  ConvertValueNodes(graph, device_context, &address_to_edge);
  std::vector<EdgePtr> graph_inputs_edges = ConvertGraphInputs(graph, device_context, &address_to_edge);

  // Convert for kernel outputs. (include graph outputs)
  const auto &nodes = graph->execution_order();
  std::vector<SingleOpPtr> single_ops;
  single_ops.reserve(nodes.size());
  for (auto const &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);

    std::vector<EdgePtr> input_edges = ConvertSingleOpInputEdges(node, address_to_edge);
    std::vector<EdgePtr> output_edges = ConvertSingleOpOutputEdges(node, device_context, &address_to_edge);

    const auto &prim = common::AnfAlgo::GetCNodePrimitive(node);
    auto single_op = std::make_unique<SingleOp>(prim, node, input_edges, output_edges);
    single_ops.push_back(std::move(single_op));
  }

  std::vector<EdgePtr> graph_outputs_edges = ConvertGraphOutputs(graph, address_to_edge);

  std::vector<EdgePtr> all_edges;
  all_edges.reserve(address_to_edge.size());
  std::transform(address_to_edge.begin(), address_to_edge.end(), std::back_inserter(all_edges),
                 [](const auto &pair) { return pair.second; });

  return std::make_unique<SimpleGraph>(name, std::move(single_ops), std::move(graph_inputs_edges),
                                       std::move(graph_outputs_edges), std::move(all_edges));
}

Edge::Edge(mindspore::pynative::EdgeType type, device::DeviceAddressPtr address,
           device::DeviceAddressPtr origin_address, session::KernelWithIndex node_with_index)
    : type_(type),
      id_(MakeEdgeId()),
      ignore_h2d_(false),
      address_(std::move(address)),
      origin_address_(origin_address),
      node_with_index_(std::move(node_with_index)) {}

SingleOp::SingleOp(PrimitivePtr primitive, CNodePtr kernel, std::vector<EdgePtr> inputs, std::vector<EdgePtr> outputs)
    : id_(MakeOpId()),
      primitive_(std::move(primitive)),
      kernel_(std::move(kernel)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)) {}

SimpleGraph::SimpleGraph(std::string name, std::vector<SingleOpPtr> single_ops, std::vector<EdgePtr> inputs,
                         std::vector<EdgePtr> outputs, std::vector<EdgePtr> all_edges)
    : name_(std::move(name)),
      single_ops_(std::move(single_ops)),
      inputs_(std::move(inputs)),
      outputs_(std::move(outputs)),
      all_edges_(std::move(all_edges)) {}

nlohmann::json Edge::DebugInfo() const {
  nlohmann::json result;
  result[kName] = kEdge + std::to_string(id_);
  result[kType] = EdgeEnumToString(type_);
  return result;
}

nlohmann::json SingleOp::DebugInfo() const {
  nlohmann::json result;
  result[kName] = kSingleOp + std::to_string(id_);
  result[kOpName] = kernel_->fullname_with_scope();

  std::vector<nlohmann::json> inputs;
  inputs.reserve(inputs_.size());
  for (const auto &input : inputs_) {
    (void)inputs.emplace_back(input->DebugInfo());
  }
  result[kInputs] = inputs;

  std::vector<nlohmann::json> outputs;
  outputs.reserve(outputs_.size());
  for (const auto &output : outputs_) {
    (void)outputs.emplace_back(output->DebugInfo());
  }
  return result;
}

nlohmann::json SimpleGraph::DebugInfo() const {
  nlohmann::json result;
  result[kName] = name_;

  std::vector<nlohmann::json> inputs;
  inputs.reserve(inputs_.size());
  for (const auto &input : inputs_) {
    (void)inputs.emplace_back(input->DebugInfo());
  }
  result[kInputs] = inputs;

  std::vector<nlohmann::json> single_ops;
  single_ops.reserve(single_ops_.size());
  for (const auto &single_op : single_ops_) {
    (void)single_ops.emplace_back(single_op->DebugInfo());
  }
  result[kSingleOp] = single_ops;

  std::vector<nlohmann::json> outputs;
  outputs.reserve(outputs_.size());
  for (const auto &output : outputs_) {
    (void)outputs.emplace_back(output->DebugInfo());
  }
  result[kOutputs] = outputs;

  return result;
}
}  // namespace pynative
}  // namespace mindspore
