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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_SRC_CUSTOM_KERNEL_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_SRC_CUSTOM_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include "src/litert/kernel/ascend/src/acl_model_options.h"
#include "src/litert/kernel/ascend/src/model_infer.h"
#include "include/api/types.h"
#include "include/api/context.h"
#include "include/api/kernel.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
namespace acl {
using mindspore::lite::STATUS;

class CustomAscendKernel : public kernel::Kernel {
 public:
  CustomAscendKernel(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                     const mindspore::schema::Primitive *primitive, const mindspore::Context *ctx);
  ~CustomAscendKernel() override;

  STATUS Prepare() override;
  STATUS ReSize() override;
  STATUS Execute() override;

 private:
  void RecordInputDataIndex();
  STATUS PrepareModelInfer();
  STATUS ProcDynamicInput(std::vector<mindspore::MSTensor> *input);
  STATUS GetRealBatchSize(std::vector<mindspore::MSTensor> *inputs, int32_t *batch_size);
  STATUS GetRealImageSize(std::vector<mindspore::MSTensor> *inputs, int32_t *image_size, int32_t num);

  bool load_model_;
  bool prepare_flag_;
  AclModelOptions acl_options_;
  std::shared_ptr<ModelInfer> model_infer_;
  size_t InputDataIndex_;
};
}  // namespace acl
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_EXTENDRT_KERNEL_ASCEND_SRC_CUSTOM_KERNEL_H_
