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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_WHERE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_WHERE_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/fp32/where_fp32.h"
#include "src/litert/kernel/cpu/base/layout_transform.h"

using mindspore::lite::InnerContext;

namespace mindspore::kernel {
class WhereCPUKernel : public LiteKernel {
 public:
  WhereCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    where_param_ = reinterpret_cast<WhereParameter *>(op_parameter_);
  }
  ~WhereCPUKernel() = default;

  int Prepare() override;
  int PreProcess() override;
  int ReSize() override { return 0; }
  int Run() override;
  virtual int DoExcute(int task_id);

 protected:
  WhereParameter *where_param_;
  bool *condition_ = nullptr;
  int32_t *int32_condition_ = nullptr;
  float *fp32_condition_ = nullptr;
  void *x_ = nullptr;
  void *y_ = nullptr;
  void *output_data_ = nullptr;

 private:
  int RunWithSingleInput();
  int RunWithTripleInputs();
};
}  // namespace mindspore::kernel
#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_WHERE_FP32_H_
