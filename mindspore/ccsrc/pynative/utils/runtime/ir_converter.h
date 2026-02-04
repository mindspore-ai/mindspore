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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_RUNTIME_IR_CONVERTER_H_
#define MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_RUNTIME_IR_CONVERTER_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"

namespace mindspore {
namespace pynative {
using KernelTensorPtr = kernel::KernelTensorPtr;
enum class EdgeType : uint8_t {
  kParameterEdge,
  kValueNodeEdge,
  kOpOutputEdge,
};

// A new simpler IR for PyNative runtime.
// Edge:
//    DeviceAddress
//
// SingleOp:
//    inputs: list[Edge]
//    outputs: list[Edge]
//
// SimpleGrpah:
//    inputs: list[Edge]
//    outputs: list[Edge]
//    SingleOps: list[SingleOp]
//
// This IR has the following three characteristics:
// 1. The same Edge contains the same KernelTensor,
//    and there is no need to sense Ref information at runtime.
// 2. The Edges of the IR graph inputs are the same as the Edges of SingleOp inputs.
//    The Edges of Graph inputs are refreshed according to the input Tensors,
//    and the correct KernelTensor is naturally obtained when SingleOp is executed.
// 3. The output Edges of SimpleGraph are the same as the output Edges of SingleOp.
//    After the operator is executed, the output Edges of Graph are automatically updated,
//    and there is no need to additionally update the outputs of Graph.
struct Edge {
  Edge(EdgeType type, KernelTensorPtr kernel_tensor, KernelTensorPtr origin_kernel_tensor,
       session::KernelWithIndex node_with_index);
  nlohmann::json DebugInfo() const;
  const EdgeType type_;
  const uint64_t id_;
  bool ignore_h2d_;
  bool is_grad_;
  KernelTensorPtr kernel_tensor_;
  // For cloning kernel tensor faster.
  const KernelTensorPtr origin_kernel_tensor_;
  const session::KernelWithIndex node_with_index_;
};
using EdgePtr = std::shared_ptr<Edge>;

// Edge1 Edge2
//   \    /
// SingleOp
//     |
//   Edge3
struct SingleOp {
  SingleOp(PrimitivePtr primitive, CNodePtr kernel, std::vector<EdgePtr> inputs, std::vector<EdgePtr> outputs);
  nlohmann::json DebugInfo() const;
  const uint64_t id_;
  const PrimitivePtr primitive_;
  const CNodePtr kernel_;
  const std::vector<EdgePtr> inputs_;
  const std::vector<EdgePtr> outputs_;
};
using SingleOpPtr = std::unique_ptr<SingleOp>;

// SimpleGraph:
//
// inputs: Edge1, Edge2
//
// Edge1  Edge2
//    \    /
//   SingleOp1
//      |
//    Edge3
//      |
//   SingleOp2
//     |
//   Edge4
//
// outputs: Edge4
struct SimpleGraph {
  SimpleGraph(std::string name, std::vector<SingleOpPtr> single_ops, std::vector<EdgePtr> inputs,
              std::vector<EdgePtr> outputs, std::vector<EdgePtr> all_edges);
  nlohmann::json DebugInfo() const;
  const std::string name_;
  const std::vector<SingleOpPtr> single_ops_;
  const std::vector<EdgePtr> inputs_;
  const std::vector<EdgePtr> outputs_;
  const std::vector<EdgePtr> all_edges_;
};
using SimpleGraphPtr = std::unique_ptr<SimpleGraph>;

// Convert ANF IR to a simpler IR
class IrConverter {
 public:
  static SimpleGraphPtr Convert(const std::string &name, const KernelGraphPtr &graph,
                                const device::DeviceContext *device_context);
};
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PYNATIVE_UTILS_RUNTIME_IR_CONVERTER_H_
