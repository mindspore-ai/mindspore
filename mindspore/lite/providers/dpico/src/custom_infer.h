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
#ifndef MINDSPORE_LITE_NNACL_CUSTOM_PARAMETER_H_
#define MINDSPORE_LITE_NNACL_CUSTOM_PARAMETER_H_
#include <vector>
#include <memory>
#include <string>
#include "include/kernel_interface.h"

namespace mindspore {
namespace dpico {
class CustomInterface : public mindspore::kernel::KernelInterface {
 public:
  CustomInterface() {}

  ~CustomInterface() = default;

  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const mindspore::schema::Primitive *primitive, const kernel::Kernel *kernel) override;

 private:
  Status InferShapeJudge(std::vector<mindspore::MSTensor> *inputs,
                         const std::vector<std::vector<int64_t>> &inputs_shape) const;
  Status InferRecurrentTwoOutputProcess(const mindspore::schema::Primitive *primitive, const kernel::Kernel *kernel,
                                        std::vector<std::vector<int64_t>> *outputs_shape) const;
};
}  // namespace dpico
}  // namespace mindspore
#endif  // MINDSPORE_LITE_NNACL_CUSTOM_PARAMETER_H_
