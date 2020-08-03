/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_LITE_SESSION_H_
#define MINDSPORE_LITE_SRC_LITE_SESSION_H_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "include/ms_tensor.h"
#include "include/lite_session.h"
#include "include/model.h"
#include "include/context.h"
#include "src/lite_kernel.h"
#include "schema/model_generated.h"

namespace mindspore {
namespace lite {
class LiteSession : public session::LiteSession {
 public:
  LiteSession() = default;

  ~LiteSession() override;

  int Init(Context *context);

  void BindThread(bool ifBind) override;

  int CompileGraph(Model *model) override;

  std::vector<mindspore::tensor::MSTensor *> GetInputs() override;

  std::vector<mindspore::tensor::MSTensor *> GetInputsByName(std::string name) override;

  int RunGraph() override;

  int RunGraph(const kernel::KernelCallBack &before = nullptr, const kernel::KernelCallBack &after = nullptr);

  std::vector<mindspore::tensor::MSTensor *> GetOutputs() override;

  std::vector<mindspore::tensor::MSTensor *> GetOutputsByName(std::string name) override;

 protected:
  int ConvertTensors(const lite::Model *model);

  void InitGraphInOutTensor(const lite::Model *model);

 protected:
  Context *context_ = nullptr;
  std::vector<kernel::LiteKernel *> kernels;
  std::vector<tensor::Tensor *> tensors;
  // graph input tensors
  std::vector<tensor::Tensor *> inputs;
  // graph output tensors
  std::vector<tensor::Tensor *> outputs;
  // graph input node name -- input tensors
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> input_map;
  // graph output node name -- output tensors
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> output_map;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITE_SESSION_H_

