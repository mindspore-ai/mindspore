/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_COMPILER_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_COMPILER_H_

#include <utility>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <set>
#include "utils/ms_utils.h"
#include "include/backend/kernel_graph.h"
#include "backend/common/session/session_basic.h"
#include "runtime/hardware/device_context.h"

namespace mindspore {
using device::DeviceContext;
using session::KernelWithIndex;
namespace pynative {
struct ExecuteKernelInfo {
  std::vector<device::DeviceAddressPtr> inputs_device_address_;
  std::vector<device::DeviceAddressPtr> outputs_device_address_;
  CNodePtr kernel_;
  PrimitivePtr primitive_;
};
using ExecuteKernelInfoList = std::vector<ExecuteKernelInfo>;

struct OpCompilerInfo {
  OpCompilerInfo(GraphInfo graph_info, GraphId graph_id, KernelGraphPtr graph, DeviceContext *device_context,
                 bool need_erase, bool need_refresh_abstract, std::vector<KernelWithIndex> graph_output_nodes,
                 std::vector<size_t> graph_outputs_tensor_num, std::vector<std::string> graph_outputs_padding_type)
      : graph_info_(std::move(graph_info)),
        graph_id_(graph_id),
        graph_(std::move(graph)),
        device_context_(device_context),
        need_erase_(need_erase),
        need_refresh_abstract_(need_refresh_abstract),
        graph_output_nodes_(std::move(graph_output_nodes)),
        graph_outputs_tensor_num_(std::move(graph_outputs_tensor_num)),
        graph_outputs_padding_type_(std::move(graph_outputs_padding_type)) {}
  ~OpCompilerInfo() = default;
  mindspore::GraphInfo graph_info_;
  GraphId graph_id_;
  KernelGraphPtr graph_;
  DeviceContext *device_context_;
  bool need_erase_;
  bool need_refresh_abstract_;
  std::vector<KernelWithIndex> graph_output_nodes_;
  std::vector<size_t> graph_outputs_tensor_num_;
  std::vector<std::string> graph_outputs_padding_type_;
  std::vector<device::DeviceAddressPtr> inputs_;
  std::vector<device::DeviceAddressPtr> outputs_;
  std::map<device::DeviceAddressPtr, tensor::TensorPtr> value_map_to_tensor_;
  ExecuteKernelInfoList execute_kernel_list_;
  std::set<device::DeviceAddressPtr> ignore_host_to_device_inputs_;
};
using OpCompilerInfoPtr = std::shared_ptr<OpCompilerInfo>;

// FuncGraph, Backend and GraphCompiler correspond one-to-one,
// and GraphCompiler stores the compilation cache of operators.
// When the graph structure changes, the front-end will send multiple graphs,
// the operators of each graph will be compiled separately, which will result in very poor performance.
// Therefore, the OpCompiler class is required to save all operator caches and make them independent of Graph.
class BACKEND_EXPORT OpCompiler {
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
  void BatchBuild(const std::vector<KernelGraphPtr> &graphs, const DeviceContext *device_context,
                  bool is_dynamic_shape = false) const;

  std::string GetSingleOpGraphInfo(const pynative::BaseOpRunInfo &op_info, const PrimitivePtr &op_prim) const;

  // Clear anf resources before process exit.
  void ClearAllCache();

  bool IsInvalidInferResultOp(const std::string &op_name) const;

 private:
  OpCompiler();
  ~OpCompiler() = default;
  DISABLE_COPY_AND_ASSIGN(OpCompiler);
  KernelGraphPtr GenerateKernelGraph(const session::BackendOpRunInfoPtr &op_run_info,
                                     const device::DeviceContext *device_context) const;

  void ConvertGraphToExecuteInfo(const OpCompilerInfoPtr &op_compiler_info) const;
  // All operators shared the same session.
  session::SessionPtr session_;
  mindspore::HashMap<mindspore::GraphInfo, OpCompilerInfoPtr> op_compiler_infos_;
};
}  // namespace pynative
using OpCompilerInfoPtr = pynative::OpCompilerInfoPtr;
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_PYNATIVE_OP_COMPILER_H_
