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
#ifndef MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_H_
#define MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "bolt/bolt_utils.h"
#include "bolt/common/uni/include/algorithm_map.h"
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel::bolt {
class BoltKernel : public LiteKernel {
 public:
  BoltKernel(const ParameterSpec &param_spec, const std::vector<lite::Tensor *> &inputs,
             const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(nullptr, inputs, outputs, ctx) {}
  ~BoltKernel() override = default;

  int Prepare() override;
  int Run() override;

 protected:
  ArchInfo arch_info_;
  BoltDataType dt_;

  std::vector<BoltTensor> bolt_in_tensors_;
  std::vector<BoltTensor> bolt_out_tensors_;
  BoltTensor tmp_tensor_;

 private:
  int InitArch();
  int UpdateTensors();
};
}  // namespace mindspore::kernel::bolt

#endif  // MINDSPORE_LITE_SRC_LITERT_KERNEL_CPU_BOLT_BOLT_KERNEL_H_
