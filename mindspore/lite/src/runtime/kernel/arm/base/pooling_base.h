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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_POOLING_BASE_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_POOLING_BASE_H_

#include <vector>
#include "src/lite_kernel.h"
#include "nnacl/fp32/pooling_fp32.h"
#include "include/errorcode.h"

using mindspore::lite::InnerContext;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
namespace mindspore::kernel {
class PoolingBaseCPUKernel : public LiteKernel {
 public:
  PoolingBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                       const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), ctx_(ctx), thread_count_(ctx->thread_num_) {
    pooling_param_ = reinterpret_cast<PoolingParameter *>(op_parameter_);
  }
  ~PoolingBaseCPUKernel() = default;

  int Init() override;
  int ReSize() override;
  int Run() override { return RET_OK; }
  int SetQuantParam();
  void FreeQuantParam();

 protected:
  const InnerContext *ctx_;
  int thread_count_;
  PoolingParameter *pooling_param_;
  QuantArg **pooling_quant_arg_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_BASE_POOLING_BASE_H_
