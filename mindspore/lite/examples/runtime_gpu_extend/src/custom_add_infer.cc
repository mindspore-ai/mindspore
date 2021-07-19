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

#include "src/custom_common.h"
#include "include/errorcode.h"
#include "include/registry/register_kernel_interface.h"

namespace mindspore {
/**
 * CustomAddInfer is a child class to infer current node output's information, including format, data_type and shape.
 * if inputs' shape exist -1, don't worry, which shows that shape will be inferred when running.
 */
class CustomAddInfer : public kernel::KernelInterface {
 public:
  CustomAddInfer() = default;
  ~CustomAddInfer() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    auto ret = custom_common::CheckInputs(*inputs);
    if (ret != lite::RET_OK) {
      if (ret == lite::RET_INFER_INVALID) {
        (*outputs)[0].SetShape({-1});  // shape{-1} shows that shape need to be inferred when running.
        return kLiteInferInvalid;
      } else {
        return kLiteError;
      }
    }
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};
std::shared_ptr<kernel::KernelInterface> CustomAddInferCreator() { return std::make_shared<CustomAddInfer>(); }
REGISTER_CUSTOM_KERNEL_INTERFACE(Tutorial, Custom_Add, CustomAddInferCreator)
}  // namespace mindspore
