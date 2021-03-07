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

#ifndef MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_
#define MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_

#include <vector>
#include "src/lite_kernel.h"
#include "include/errorcode.h"
#include "schema/model_generated.h"
#include "src/runtime/kernel/arm/base/convolution_base.h"
#include "nnacl/int8/conv1x1_int8.h"
#include "nnacl/base/conv1x1_base.h"
#include "nnacl/int8/matmul_int8.h"
#include "nnacl/matmul_parameter.h"
#include "src/common/utils.h"

namespace mindspore::kernel {
class Convolution1x1Int8CPUKernel : public ConvolutionBaseCPUKernel {
 public:
  Convolution1x1Int8CPUKernel(OpParameter *parameter, const std::vector<lite::Tensor *> &inputs,
                              const std::vector<lite::Tensor *> &outputs, const InnerContext *ctx)
      : ConvolutionBaseCPUKernel(parameter, inputs, outputs, ctx) {}
  ~Convolution1x1Int8CPUKernel() override;

  int Init() override;
  int ReSize() override;
  int Run() override;

 private:
  int InitRunBuf();
  void FreeRunBuf();

 public:
  int OcRun(int task_id);
  int HwRun(int task_id);
  int OcOptPre(int task_id);

 private:
  int RunArmOc(int task_id);
  int RunArm64OptOc(int task_id);
  int RunArmHw(int task_id);
  int RunArm64OptHw(int task_id);

 private:
  void FreeResizeBuf();
  int InitParam();
  int InitWeightBias();
  int InitWeightBiasArm32();
  void Pre1x1Trans(int8_t *src_input, int8_t *src_output);
  void CheckSupportOptimize();
  int InitBiasByzp(const void *src_weight, int input_channel, int output_channel, int round_oc);

 private:
  int32_t *input_sum_ = nullptr;     /* per-oc */
  int32_t *filter_zp_ptr_ = nullptr; /* per-oc up round  */
  int32_t *left_shift_ = nullptr;    /* per-oc up round  */
  int32_t *right_shift_ = nullptr;   /* per-oc up round  */
  int32_t *multiplier_ = nullptr;    /* per-oc up round  */
  int8_t *packed_weight_ = nullptr;
  int8_t *packed_input_ = nullptr;
  int8_t *input_ptr_ = nullptr;
  int8_t *output_ptr_ = nullptr;
  size_t thread_count_hw_ = 1;
  size_t thread_stride_hw_ = 0;
  size_t thread_count_oc_ = 1;
  size_t thread_stride_oc_ = 0;
  bool pre_trans_input_ = false;
  bool parallel_by_oc_ = false;
  size_t input_sum_size_ = 0;
  MatMulParameter *matmul_param_ = nullptr;
  MATMUL_OPT_DP_FUNC matmul_func_ = nullptr;
  bool support_optimize_ = false;
  bool filter_peroc_ = false;
};
}  // namespace mindspore::kernel

#endif  // MINDSPORE_LITE_SRC_RUNTIME_KERNEL_ARM_INT8_CONVOLUTION_1x1_INT8_H_
