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

#include "include/api/status.h"
#include "include/registry/register_kernel_interface.h"

namespace mindspore {
class CustomOMInfer : public kernel::KernelInterface {
 public:
  CustomOMInfer() = default;
  ~CustomOMInfer() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {
    return kSuccess;
  }
};
std::shared_ptr<kernel::KernelInterface> CustomOMInferCreator() { return std::make_shared<CustomOMInfer>(); }
REGISTER_CUSTOM_KERNEL_INTERFACE(Tutorial, Custom_OM, CustomOMInferCreator)
}  // namespace mindspore
