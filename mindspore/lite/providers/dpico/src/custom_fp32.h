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

#ifndef MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_FP32_H_
#define MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_FP32_H_

#include <vector>
#include <string>
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "manager/acl_model_manager.h"

using mindspore::kernel::Kernel;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore {
namespace lite {
class CustomCPUKernel : public Kernel {
 public:
  CustomCPUKernel(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                  const mindspore::schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}

  ~CustomCPUKernel() = default;

  int Prepare() override;
  int ReSize() override;
  int Execute() override;

 private:
  int ParseAttrs();
  bool InferShapeDone() const;
  int PreProcess();
  void UpdateInputOutputTensor();

 private:
  AclModelManagerPtr acl_model_manager_{nullptr};
  std::string support_zero_copy_{"off"};
  size_t stride_align_size_{16};
  size_t custom_id_{0};
  size_t custom_num_{0};
};
}  // namespace lite
}  // namespace mindspore

#endif  // MINDSPORE_LITE_PROVIDERS_DPICO_SRC_CUSTOM_FP32_H_
