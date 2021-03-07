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

#ifndef MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_TANH_INT8_H_
#define MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_TANH_INT8_H_

#include <vector>
#include <limits>
#include <algorithm>
#include "src/lite_kernel.h"
#include "nnacl/int8/tanh_int8.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
#include "include/errorcode.h"

namespace mindspore::kernel {
class TanhInt8CPUKernel : public LiteKernel {
 public:
  TanhInt8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                    const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx) {}
  ~TanhInt8CPUKernel() override = default;

  int Init() override;
  int ReSize() override;
  int Run() override;

 public:
  int DoActivation(int task_id);

 private:
  int8_t *in_ptr_;
  int8_t *out_ptr_;
  int element_size_;
  int thread_count_;
  int thread_stride_;
  TanhQuantParameter tanh_quant_;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_BACKEND_ARM_INT8_TANH_INT8_H_
