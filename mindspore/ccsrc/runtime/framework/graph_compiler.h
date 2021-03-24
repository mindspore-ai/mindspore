/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include "runtime/hardware/device_context.h"
#include "backend/session/session_basic.h"

namespace mindspore {
namespace runtime {
class GraphCompiler {
 public:
  static GraphCompiler &GetInstance() {
    static GraphCompiler instance;
    return instance;
  }

  // Set device context which is initialized, the function must be called
  // before using GraphCompiler and after changing device type or device id.
  void set_device_context(device::DeviceContext *device_context);

  // Construct kernel graph from anf nodes list and compile kernel graph in Graph mode,
  // the detailed implementation of compiling graph is in 'CompileGraphImpl'.
  GraphId CompileGraph(const AnfNodePtrList &nodes, const AnfNodePtrList &outputs);

  // Run a graph and get the output in Graph mode.
  void RunGraph(const GraphId &graph_id, const std::vector<tensor::TensorPtr> &inputs, VectorRef *outputs);

  // Construct single op kernel graph, compile and run the kernel graph in PyNative mode.
  void CompileAndRunGraph(session::OpRunInfo *op_run_info, const GraphInfo &graph_info,
                          std::vector<tensor::TensorPtr> *input_tensors, const std::vector<int64_t> &tensors_mask,
                          VectorRef *outputs);

 private:
  GraphCompiler() = default;
  ~GraphCompiler() = default;
  DISABLE_COPY_AND_ASSIGN(GraphCompiler);

  // The implementation of compiling graph in Graph Mode, including optimizing graph,
  // setting operator info, creating kernel and transforming kernel graph to ActorSet.
  GraphId CompileGraphImpl(const KernelGraphPtr &graph);

  device::DeviceContext *device_context_{nullptr};

  // Single op kernel graph cache for PyNative mode.
  std::unordered_map<GraphInfo, KernelGraphPtr> run_op_graphs_;

  // The member variable 'session_' will be removed after removing session module.
  session::SessionPtr session_{nullptr};
};
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_COMPILER_H_
