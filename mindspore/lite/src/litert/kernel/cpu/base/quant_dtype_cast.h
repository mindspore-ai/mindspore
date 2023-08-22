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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_QUANT_DTYPE_CAST_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_QUANT_DTYPE_CAST_H_

#include <vector>
#include "src/litert/lite_kernel.h"

namespace mindspore::kernel {
class QuantDTypeCastCPUKernel : public LiteKernel {
 public:
  QuantDTypeCastCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                          const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx)
      : LiteKernel(parameter, inputs, outputs, ctx), thread_num_(ctx->thread_num_) {}
  ~QuantDTypeCastCPUKernel() override {
    if (scale_ != nullptr) {
      free(scale_);
      scale_ = nullptr;
    }
    if (zero_point_ != nullptr) {
      free(zero_point_);
      zero_point_ = nullptr;
    }
  };

  int Prepare() override;
  int ReSize() override;
  int Run() override;
  int QuantDTypeCast(int task_id);

 private:
  int DoDequantInt8ToFp32(int num_unit_thread, int thread_offset, int channel_offset);
  int DoDequantFp32ToInt8(int num_unit_thread, int thread_offset);
  int ExtractQuantParams(const lite::Tensor *tensor, int preferred_dim);
  int DoDequanInt8ToFp32ChannelRow(const int8_t *quant_values, float *real_values, float *scale, int32_t *zp,
                                   int channel_unit_num, int per_channel_size);
  int DoDequanInt8ToFp32ChannelCol(const int8_t *quant_values, float *real_values, float *scale, int32_t *zp,
                                   int channel_num, int channel_unit_num, int per_channel_size);

  int thread_num_;
  int thread_n_num_{0};
  int thread_n_stride_{0};
  int num_unit_{0};
  int8_t *int8_ptr_ = nullptr;
  int8_t *int8_out_ptr_ = nullptr;
  uint8_t *uint8_ptr_ = nullptr;
  float *float32_ptr_ = nullptr;
  float *scale_ = nullptr;
  int *zero_point_ = nullptr;
  bool perchannel_ = false;
  int channel_num_{0};
  int per_channel_size_{0};
  int thread_channel_stride_{0};
  int preferred_dim_{0};

  int32_t src_dtype{0};
  int32_t dst_dtype{0};
  int32_t quant_dst_dtype{0};
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_BASE_QUANT_DTYPE_CAST_H_
