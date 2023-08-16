/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>
#include <set>
#include "plugin/device/ascend/hal/hardware/ascend_deprecated_interface.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/memory_manager.h"
#include "utils/ms_context.h"
#include "include/transform/graph_ir/types.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm_lib.h"
#include "plugin/device/ascend/hal/hardware/ge_device_res_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
struct GeInputData {
  std::vector<GeTensor> ge_inputs;
  std::vector<std::pair<AnfNodePtr, size_t>> need_update_input;
};

struct GeOutputData {
  std::vector<GeTensor> ge_outputs;
  std::vector<std::pair<AnfNodePtr, size_t>> graph_outputs;
};

class GeGraphExecutor : public GraphExecutor {
 public:
  ~GeGraphExecutor() override = default;
  bool CompileGraph(const FuncGraphPtr &graph, const std::map<string, string> &compile_options) override;
  bool RunGraph(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs,
                std::vector<tensor::Tensor> *outputs, const std::map<string, string> &compile_options) override;

  static FuncGraphPtr BuildDFGraph(const FuncGraphPtr &anf_graph, const transform::TensorOrderMap &init_inputs_map,
                                   bool export_air);
  void PreprocessBeforeRun(const KernelGraphPtr &graph);
  size_t GetGraphFeatureMemory(const FuncGraphPtr &graph) const override;

 private:
  bool RunGraphRefMode(const FuncGraphPtr &graph, const std::vector<tensor::Tensor> &inputs) const;
  void AllocInputHostMemory(const KernelGraphPtr &kernel_graph) const;
  void AllocOutputHostMemory(const KernelGraphPtr &kernel_graph) const;
  void AllocConstMemory(const transform::RunOptions &options, const KernelGraphPtr &graph, size_t memory_size) const;
  void AllocFeatureMemory(const transform::RunOptions &options, size_t memory_size) const;
  void AllocParameterMemory(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *memo = nullptr) const;
  void BuildInputDataGeTensor(const KernelGraphPtr &kernel_graph);
  void BuildOutputDataGeTensor(const KernelGraphPtr &kernel_graph);
  void AllocOutputMemory(const KernelGraphPtr &kernel_graph) const;
  bool CompileGraph(const KernelGraphPtr &graph, const std::map<string, string> &compile_options);
  std::vector<GeTensor> GenerateInputGeTensor(const KernelGraphPtr &kernel_graph) const;
  std::vector<GeTensor> GenerateOutputGeTensor(const KernelGraphPtr &kernel_graph) const;
  GeDeviceResManager *ResManager() const;
  void RunInitGraph(const std::string &graph_name) const;

  mindspore::HashMap<KernelGraphPtr, GeInputData> input_datas_;
  mindspore::HashMap<KernelGraphPtr, GeOutputData> output_datas_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_ASCEND_GE_GRAPH_EXECUTOR_H_
