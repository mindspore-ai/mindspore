/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_INFER_CUSTOM_INFER_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_INFER_CUSTOM_INFER_H_

#include <vector>
#include "include/kernel_interface.h"

namespace mindspore {
namespace dpico {
class CustomInterface : public kernel::KernelInterface {
 public:
  CustomInterface() {}

  ~CustomInterface() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive, const kernel::Kernel *kernel) override;
};
}  // namespace dpico
}  // namespace mindspore

#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_INFER_CUSTOM_INFER_H_
