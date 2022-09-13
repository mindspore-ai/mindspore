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
/// \brief Single Op Session implementation, used in Ascend Device Context.
class SingleOpInferSession : public InferSession {
 public:
  SingleOpInferSession() = default;
  ~SingleOpInferSession() override;
  Status Init(const std::shared_ptr<Context> &context) override;
  Status AscendInit(const std::shared_ptr<Context> &context);
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
  Status ResizeGraphInputs(const std::vector<tensor::Tensor> &inputs, const std::vector<std::vector<int64_t>> &dims);

  KernelGraphUtilsPtr kernel_graph_utils_;
  KernelGraphPtr kernel_graph_;
  std::vector<MutableTensorImplPtr> inputs_;
  std::vector<std::string> input_names_;
  std::vector<MutableTensorImplPtr> outputs_;
  std::vector<std::string> output_names_;
  uint32_t device_id_ = 0;
};
}  // namespace mindspore

#endif  // MINDSPORE_LITE_EXTENDRT_SINGLE_OP_SESSION_H_
