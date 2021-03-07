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
#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_LAYERNORM_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_LAYERNORM_INT8_H_

#include <limits>
#include <vector>
#include "nnacl/int8/layer_norm_int8.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
class LayerNormInt8CPUKernel : public LiteKernel {
 public:
  LayerNormInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                         const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {
    param_ = reinterpret_cast<LayerNormParameter *>(parameter);
  }
  ~LayerNormInt8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int DoExecute(int task_id);

 private:
  int SetQuantArgs();

 private:
  LayerNormParameter *param_ = nullptr;
  LayerNormQuantArg quant_param_;
  int8_t *src_ptr_ = nullptr;
  int8_t *dst_ptr_ = nullptr;
  float *gamma_ptr_ = nullptr;
  float *beta_ptr_ = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_LAYERNORM_INT8_H_
