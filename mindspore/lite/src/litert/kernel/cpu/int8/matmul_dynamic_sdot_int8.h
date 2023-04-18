/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_SDOT_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_SDOT_INT8_H_

#include <vector>
#include "src/litert/kernel/cpu/int8/matmul_dynamic_base_int8.h"

namespace mindspore::kernel {
class MatMulDynamicSdotInt8Kernel : public MatmulDynamicBaseInt8CPUKernel {
 public:
  MatMulDynamicSdotInt8Kernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : MatmulDynamicBaseInt8CPUKernel(parameter, inputs, outputs, ctx) {
    param_->matmul_type_ = MatmulType::kMatmulDynamicSdotInt8Cpu;
  }
  ~MatMulDynamicSdotInt8Kernel() override = default;
  int Run() override;

 public:
  int MatMulDynamicArm64SdotPre(int task_id);
  int MatMulDynamicArm64SdotImpl(int task_id);

 protected:
  void InitParameter() override;

 private:
  template <typename T>
  using DynamicMatmulComputer = void (*)(const int8_t *a, const int8_t *b, T *out, size_t deep4,
                                         const float *multi_scles, const T *bias, size_t row, size_t col, size_t stride,
                                         const int32_t *a_sums, const int32_t *b_sums, int64_t a_zp, int64_t b_zp_sum,
                                         int64_t act_type, int64_t mode);

  int MatMulDynamicRunArm64Sdot();
  void ComputeMultiScaleAhead(std::vector<float> *multi_scale, int col_start, size_t col_num);
  void ComputeMultiScaleChannelByChannel(std::vector<float> *multi_scale, int row_start, size_t row_num, int col_start,
                                         size_t col_num);
  int *batch_sums_ = nullptr;
  DynamicMatmulComputer<float> dynamic_matmul_compute_fp32{nullptr};
#ifdef ENABLE_FP16
  DynamicMatmulComputer<float16_t> dynamic_matmul_compute_fp16{nullptr};
#endif
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_INT8_MATMUL_DYNAMIC_SDOT_INT8_H_
