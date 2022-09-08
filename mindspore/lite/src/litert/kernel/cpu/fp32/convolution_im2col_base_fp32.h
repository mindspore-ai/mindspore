/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_IM2COL_BASE_FP32_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_IM2COL_BASE_FP32_H_

#include <vector>
#include "src/litert/lite_kernel.h"
#include "nnacl/op_base.h"
#include "src/litert/kernel/cpu/base/convolution_base.h"

namespace mindspore::kernel {
using RowMajor2ColNMajorFunc = void (*)(const float *src_ptr, float *dst_ptr, int row, int col);

int ConvolutionIm2ColImpl(void *cdata, int task_id, float lhs_scale, float rhs_scale);
class ConvolutionIm2ColBaseCPUKernel : public ConvolutionBaseCPUKernel {
 public:
  ConvolutionIm2ColBaseCPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                                 const std::vector<lite::Tensor *> &outputs, const lite::InnerContext *ctx,
                                 float *origin_weight, float *origin_bias)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx, origin_weight, origin_bias) {}
  ~ConvolutionIm2ColBaseCPUKernel() override {}

  virtual void InitGlobalVariable();
  int Prepare() override;
  virtual int InitTmpBuffer();
  int ReSize() override;
  int Run() override;
  virtual int RunImpl(int task_id);

 protected:
  int MallocWeightBiasData() override;
  void PackWeight() override;
  void FreeTmpBuffer() {
    if (packed_input_ != nullptr) {
      ctx_->allocator->Free(packed_input_);
      packed_input_ = nullptr;
    }
    if (col_major_input_ != nullptr) {
      ctx_->allocator->Free(col_major_input_);
      col_major_input_ = nullptr;
    }
    if (output_need_align_ && tmp_output_ != nullptr) {
      ctx_->allocator->Free(tmp_output_);
      tmp_output_ = nullptr;
      output_need_align_ = false;
    }
  }
  int UpdateThreadNumProcess(int32_t kernel_type, int64_t per_unit_load_num, int64_t per_unit_store_num,
                             int64_t unit_num) override;

 protected:
  float *tmp_output_ = nullptr;
  float *packed_input_ = nullptr;
  float *col_major_input_ = nullptr;
  bool output_need_align_ = false;

  int oc_tile_ = C8NUM;    // oc tile is C8NUM in C
  int row_tile_ = C12NUM;  // oc tile is C12NUM in C
  RowMajor2ColNMajorFunc rowMajor2ColNMajorFunc = nullptr;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_CPU_FP32_CONVOLUTION_IM2COL_BASE_FP32_H_
