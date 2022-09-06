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
namespace mindspore {
// TODO(zhaizhiqiang): use GraphSinkDelegateSession instead of GraphSinkSession in future.
// class GraphSinkDelegateSession
class GraphSinkSession : public InferSession {
 public:
  GraphSinkSession() = default;
  explicit GraphSinkSession(const std::shared_ptr<device::GraphExecutor> &executor) : graph_executor_(executor) {}
  virtual ~GraphSinkSession() = default;

  Status Init(const std::shared_ptr<Context> &context) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) override;
  std::vector<MutableTensorImplPtr> GetOutputs() override;
  std::vector<MutableTensorImplPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  MutableTensorImplPtr GetOutputByTensorName(const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(const std::string &name) override;

 private:
  std::shared_ptr<device::GraphExecutor> graph_executor_;
  std::map<string, string> options_;
  KernelGraphUtilsPtr kernel_graph_utils_;
  KernelGraphPtr kernel_graph_;
  std::vector<tensor::TensorPtr> inputs_;
  std::vector<std::string> input_names_;
  std::vector<tensor::TensorPtr> outputs_;
  std::vector<std::string> output_names_;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_DELEGATE_SESSION_H_
