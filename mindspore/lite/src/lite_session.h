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

  void BindThread(bool if_bind) override;

  int CompileGraph(Model *model) override;

  std::vector<mindspore::tensor::MSTensor *> GetInputs() const override;

  std::vector<mindspore::tensor::MSTensor *> GetInputsByName(const std::string &name) const override;

  int RunGraph(const session::KernelCallBack &before = nullptr,
               const session::KernelCallBack &after = nullptr) override;

  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> GetOutputs() const override;

  std::vector<mindspore::tensor::MSTensor *> GetOutputsByName(const std::string &name) const override;

 protected:
  int ConvertTensors(const lite::Model *model);

  void InitGraphInOutTensors(const lite::Model *model);
  // init this->inputs_
  void InitGraphInputTensors(const lite::Model *model);
  // init this->input_vec_
  void InitGraphInputMSTensors(const lite::Model *model);
  // init this->outputs_
  void InitGraphOutputTensors(const lite::Model *model);
  // init this->input_map_
  void InitGraphInputMap(const lite::Model *model);
  // init this->output_map_
  void InitGraphOutputMap(const lite::Model *model);

 protected:
  Context *context_ = nullptr;
  std::vector<kernel::LiteKernel *> kernels_;
  std::vector<tensor::Tensor *> tensors_;
  // graph input tensors
  std::vector<tensor::Tensor *> inputs_;
  // graph output tensors
  std::vector<tensor::Tensor *> outputs_;
  // graph input MSTensors
  std::vector<mindspore::tensor::MSTensor *> input_vec_;
  // graph input node name -- input tensors
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> input_map_;
  // graph output node name -- output tensors
  std::unordered_map<std::string, std::vector<mindspore::tensor::MSTensor *>> output_map_;
};
}  // namespace lite
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_LITE_SESSION_H_
