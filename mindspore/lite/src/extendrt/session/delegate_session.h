/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "extendrt/utils/kernel_graph_utils.h"
#include "extendrt/session/lite_graph_executor.h"
#include "extendrt/delegate/ascend_ge/ge_device_context.h"
namespace mindspore {
// TODO(zhaizhiqiang): use GraphSinkDelegateSession instead of GraphSinkSession in future.
// class GraphSinkDelegateSession
class GraphSinkSession : public InferSession {
 public:
  GraphSinkSession() = default;
  explicit GraphSinkSession(std::shared_ptr<device::GraphExecutor> graph_executor) {
    graph_executor_ = std::dynamic_pointer_cast<mindspore::LiteGraphExecutor>(graph_executor);
  }
  ~GraphSinkSession() override;

  Status Init(const std::shared_ptr<Context> &context) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                  const MSKernelCallBack &before, const MSKernelCallBack &after) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) override;
  Status Resize(const std::vector<tensor::Tensor> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<MutableTensorImplPtr> GetOutputs() override;
  std::vector<MutableTensorImplPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  MutableTensorImplPtr GetOutputByTensorName(const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(const std::string &name) override;

 private:
  Status GeDeviceContextInit();

  std::shared_ptr<mindspore::LiteGraphExecutor> graph_executor_;
  std::map<std::string, std::string> options_;
  bool is_use_kernel_graph_ = true;
  KernelGraphUtilsPtr kernel_graph_utils_;
  std::shared_ptr<Context> context_;
  KernelGraphPtr kernel_graph_;
  FuncGraphPtr func_graph_;
  std::vector<MutableTensorImplPtr> inputs_;
  std::vector<std::string> input_names_;
  std::vector<MutableTensorImplPtr> outputs_;
  std::vector<std::string> output_names_;
  std::shared_ptr<GeDeviceContext> ge_context_;

  Status InitGraphInputsOutputs();
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_DELEGATE_SESSION_H_
