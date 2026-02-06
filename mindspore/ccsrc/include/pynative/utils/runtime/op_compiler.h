/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_COMPILER_H_
#define MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_COMPILER_H_

#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <set>
#include "utils/ms_utils.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "include/backend/common/kernel_graph/session_basic.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "pynative/utils/runtime/ir_converter.h"

namespace mindspore {
using device::DeviceContext;
using session::KernelWithIndex;
using GetGraphInfoFunc = std::function<std::string(const pynative::BaseOpRunInfo &op_info, const PrimitivePtr &op_prim,
                                                   const std::string &graph_info)>;
namespace pynative {
constexpr size_t kAlignSize = 64;
struct OpCompilerInfo {
  OpCompilerInfo(GraphInfo graph_info, GraphId graph_id, KernelGraphPtr graph, DeviceContext *device_context,
                 bool need_erase, bool need_refresh_abstract, std::vector<KernelWithIndex> graph_output_nodes,
                 std::vector<size_t> graph_outputs_tensor_num, std::vector<std::string> graph_outputs_padding_type,
                 SimpleGraphPtr simple_graph)
      : graph_info_(std::move(graph_info)),
        graph_id_(graph_id),
        graph_(std::move(graph)),
        device_context_(device_context),
        need_erase_(need_erase),
        need_refresh_abstract_(need_refresh_abstract),
        graph_output_nodes_(std::move(graph_output_nodes)),
        graph_outputs_tensor_num_(std::move(graph_outputs_tensor_num)),
        graph_outputs_padding_type_(std::move(graph_outputs_padding_type)),
        simple_graph_(std::move(simple_graph)) {}
  ~OpCompilerInfo() = default;
  void UpdateStatus(bool ready);
  void WaitReady() const;
  const mindspore::GraphInfo graph_info_;
  const GraphId graph_id_;
  const KernelGraphPtr graph_;
  const DeviceContext *device_context_;
  const bool need_erase_;
  const bool need_refresh_abstract_;
  const std::vector<KernelWithIndex> graph_output_nodes_;
  const std::vector<size_t> graph_outputs_tensor_num_;
  const std::vector<std::string> graph_outputs_padding_type_;
  const SimpleGraphPtr simple_graph_;
  alignas(kAlignSize) std::atomic<bool> ready_{true};
};
using OpCompilerInfoPtr = std::shared_ptr<OpCompilerInfo>;

// FuncGraph, Backend and GraphCompiler correspond one-to-one,
// and GraphCompiler stores the compilation cache of operators.
// When the graph structure changes, the front-end will send multiple graphs,
// the operators of each graph will be compiled separately, which will result in very poor performance.
// Therefore, the OpCompiler class is required to save all operator caches and make them independent of Graph.
class PYNATIVE_UTILS_EXPORT OpCompiler {
 public:
  static OpCompiler &GetInstance();

  // Compile RunOpInfo into a KernelGraph.
  OpCompilerInfoPtr Compile(const session::BackendOpRunInfoPtr &op_run_info, bool *single_op_cache_hit,
                            const std::string &device_name, const uint32_t &device_id);

  // Clear op cache in dynamic scenes.
  // Otherwise, the operator cache will keep growing, resulting in insufficient memory.
  void ClearOpCache(const mindspore::GraphInfo &graph_info);

  // Accumulate a certain number of operators,
  // and then compile the operators in parallel to improve compilation efficiency.
  void KernelBuild(const OpCompilerInfoPtr &op_compiler_info, const DeviceContext *device_context,
                   bool is_dynamic_shape = false) const;

  std::string GetSingleOpGraphInfo(const pynative::BaseOpRunInfo &op_info, const PrimitivePtr &op_prim) const;

  // Clear anf resources before process exit.
  void ClearAllCache();

  bool IsInvalidInferResultOp(const std::string &op_name) const;

  static void UpdateRefNodeOutputDeviceAddress(const KernelGraphPtr &graph);
  void set_get_graph_info_func(const GetGraphInfoFunc &func) { get_graph_info_func_ = func; }

 private:
  OpCompiler();
  ~OpCompiler() = default;
  DISABLE_COPY_AND_ASSIGN(OpCompiler);
  KernelGraphPtr GenerateKernelGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                     const device::DeviceContext *device_context) const;
  void AssignStreamIdForSingleOpGraph(const KernelGraphPtr &graph, uint32_t stream_id);
  // All operators shared the same session.
  session::SessionPtr session_;
  mindspore::HashMap<mindspore::GraphInfo, OpCompilerInfoPtr> op_compiler_infos_;
  GetGraphInfoFunc get_graph_info_func_{nullptr};
};
}  // namespace pynative
using OpCompilerInfoPtr = pynative::OpCompilerInfoPtr;
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_INCLUDE_PYNATIVE_UTILS_RUNTIME_OP_COMPILER_H_
