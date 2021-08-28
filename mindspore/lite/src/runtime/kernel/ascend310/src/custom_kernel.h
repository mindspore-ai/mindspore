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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_

#include <vector>
#include <memory>
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/ascend310/src/model_infer.h"

using mindspore::lite::STATUS;

namespace mindspore {
namespace acl {
class CustomAscend310Kernel : public kernel::Kernel {
 public:
  CustomAscend310Kernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                        const mindspore::schema::Primitive *primitive, const mindspore::Context *ctx);
  ~CustomAscend310Kernel() override;

  STATUS Prepare() override;
  STATUS ReSize() override;
  STATUS Execute() override;

 private:
  STATUS PrepareModelInfer();

  bool load_model_;
  std::shared_ptr<ModelInfer> model_infer_;
};
}  // namespace acl
}  // namespace mindspore

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND310_FP32_CUSTOM_H_
