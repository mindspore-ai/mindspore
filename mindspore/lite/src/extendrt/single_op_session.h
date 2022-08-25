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
#ifndef MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_

#include <string>
#include <memory>
#include <vector>

#include "src/extendrt/infer_session.h"
#include "extendrt/utils/kernel_graph_utils.h"

namespace mindspore {
class SingleOpInferSession : public InferSession {
 public:
  SingleOpInferSession() = default;
  virtual ~SingleOpInferSession() = default;
  Status Init(const std::shared_ptr<Context> context) override;
  Status AscendInit(const std::shared_ptr<Context> &context);
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) override;
  Status RunGraph() override;
  Status RunGraph(const std::vector<tensor::TensorPtr> &inputs, std::vector<tensor::TensorPtr> *outputs) override;
  Status Resize(const std::vector<tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims) override;

  std::vector<tensor::TensorPtr> GetOutputs() override;
  std::vector<tensor::TensorPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  tensor::TensorPtr GetOutputByTensorName(const std::string &tensorName) override;
  tensor::TensorPtr GetInputByTensorName(const std::string &name) override;

 private:
  Status ResizeGraphInputs(const std::vector<tensor::TensorPtr> &inputs, const std::vector<std::vector<int64_t>> &dims);

  KernelGraphUtilsPtr kernel_graph_utils_;
  KernelGraphPtr kernel_graph_;
  std::vector<tensor::TensorPtr> inputs_;
  std::vector<std::string> input_names_;
  std::vector<tensor::TensorPtr> outputs_;
  std::vector<std::string> output_names_;
  uint32_t device_id_ = 0;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_
