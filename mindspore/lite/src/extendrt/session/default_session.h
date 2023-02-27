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
#ifndef MINDSPORE_LITE_EXTENDRT_SESSION_DEFAULT_SESSION_H_
#define MINDSPORE_LITE_EXTENDRT_SESSION_DEFAULT_SESSION_H_

#include <vector>
#include <string>
#include <memory>
#include <map>

#include "extendrt/infer_session.h"

#include "infer/graph_compiler.h"
#include "infer/graph_runtime.h"

namespace mindspore {
/// \brief Default Infer Session Implementation, using kernelmod, not implemented now.
class DefaultInferSession : public InferSession {
 public:
  explicit DefaultInferSession(const std::shared_ptr<Context> &context) { context_ = context; }
  virtual ~DefaultInferSession() = default;
  Status Init(const std::shared_ptr<Context> &context) override;
  Status CompileGraph(FuncGraphPtr graph, const void *data = nullptr, size_t size = 0) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) override;
  Status RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                  const MSKernelCallBack &before, const MSKernelCallBack &after) override;
  Status Resize(const std::vector<tensor::Tensor> &inputs, const std::vector<std::vector<int64_t>> &dims) override;
  std::vector<MutableTensorImplPtr> GetOutputs() override;
  std::vector<MutableTensorImplPtr> GetInputs() override;
  std::vector<std::string> GetOutputNames() override;
  std::vector<std::string> GetInputNames() override;
  MutableTensorImplPtr GetOutputByTensorName(const std::string &tensorName) override;
  MutableTensorImplPtr GetInputByTensorName(const std::string &name) override;

 protected:
  virtual std::shared_ptr<infer::abstract::GraphCompiler> GetGraphCompiler() { return compiler_; }

  virtual std::shared_ptr<infer::abstract::GraphRuntime> GetGraphRuntime() { return runtime_; }

 private:
  Status CopyDataToInnerTensors(const std::vector<tensor::Tensor> &tensors,
                                std::vector<infer::abstract::Tensor *> inner_tensors);
  std::vector<MutableTensorImplPtr> AbstractTensorsToTensorImpls(
    const std::vector<infer::abstract::Tensor *> &abstract_tensors);
  std::vector<mindspore::tensor::Tensor> LiteTensorToTensor(
    const std::vector<infer::abstract::Tensor *> &abstract_tensors);
  std::vector<int32_t> TruncateShape(const std::vector<int64_t> &shape, enum TypeId type, size_t data_len,
                                     bool verify_size);

 private:
  std::shared_ptr<infer::abstract::GraphCompiler> compiler_;

  std::shared_ptr<infer::abstract::GraphRuntime> runtime_;

  const std::shared_ptr<Context> &context_;
};
}  // namespace mindspore
#endif  // MINDSPORE_LITE_EXTENDRT_SESSION_DEFAULT_SESSION_H_
