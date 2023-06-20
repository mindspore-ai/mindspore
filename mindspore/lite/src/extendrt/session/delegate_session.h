/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_DELEGATE_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_DELEGATE_SESSION_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "extendrt/infer_session.h"
#include "runtime/hardware/device_context.h"
#include "extendrt/session/lite_graph_executor.h"

namespace mindspore {
/// \brief Delegate Session implementation, use delegate api for inference.
// (zhaizhiqiang): use GraphSinkDelegateSession instead of GraphSinkSession in future.
// class GraphSinkDelegateSession
struct DelegateGraphInfo {
  std::vector<MutableTensorImplPtr> inputs;
  std::vector<std::string> input_names;
  std::vector<MutableTensorImplPtr> outputs;
  std::vector<std::string> output_names;
};

class GraphSinkSession : public InferSession {
 public:
  GraphSinkSession() = default;
  explicit GraphSinkSession(std::shared_ptr<device::GraphExecutor> graph_executor) {
    graph_executor_ = std::dynamic_pointer_cast<mindspore::LiteGraphExecutor>(graph_executor);
  }
  ~GraphSinkSession() override;

  Status Init(const std::shared_ptr<Context> &context, const ConfigInfos &config_info = {}) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0,
                      uint32_t *graph_id = nullptr) override;
  Status CompileGraph(const void *model_data, size_t data_size, uint32_t *graph_id) override;
  Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                  const MSKernelCallBack &before, const MSKernelCallBack &after) override;
  Status RunGraph(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                  std::vector<tensor::Tensor> *outputs) override;
  Status Resize(uint32_t graph_id, const std::vector<tensor::Tensor> &inputs,
                const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<MutableTensorImplPtr> GetOutputs(uint32_t graph_id) override;
  std::vector<MutableTensorImplPtr> GetInputs(uint32_t graph_id) override;
  std::vector<std::string> GetOutputNames(uint32_t graph_id) override;
  std::vector<std::string> GetInputNames(uint32_t graph_id) override;
  MutableTensorImplPtr GetOutputByTensorName(uint32_t graph_id, const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(uint32_t graph_id, const std::string &name) override;
  void SetConfigInfo(ConfigInfos config_infos) { config_infos_ = config_infos; }

 private:
  Status InitGraphInfo(DelegateGraphInfo *graph_info_ptr, uint32_t graph_id);
  Status InitGraphInputsOutputs(const FuncGraphPtr &graph, DelegateGraphInfo *graph_info);
  Status UpdateGraphInputsOutputs(uint32_t graph_id, DelegateGraphInfo *graph_info);
  void UpdateDataFlowGraphInputsOutputs(DelegateGraphInfo *graph_info_ptr, const std::vector<tensor::Tensor> &inputs,
                                        const std::vector<tensor::Tensor> &outputs);

  std::shared_ptr<mindspore::LiteGraphExecutor> graph_executor_;
  std::map<std::string, std::string> options_;
  std::map<uint32_t, DelegateGraphInfo> graph_infos_;
  bool is_data_flow_graph_ = false;
  std::shared_ptr<Context> context_;
  ConfigInfos config_infos_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_DELEGATE_SESSION_H_
