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
#include "src/runtime/kernel/ascend310/src/acl_model_options.h"
#include "src/runtime/kernel/ascend310/src/model_infer.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
using mindspore::lite::STATUS;

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
  AclModelOptions GetAclModelOptions(const mindspore::Context *ctx) const;

  bool load_model_;
  std::shared_ptr<ModelInfer> model_infer_;
};
}  // namespace acl
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ASCEND310_KERNEL_CUSTOM_H_
