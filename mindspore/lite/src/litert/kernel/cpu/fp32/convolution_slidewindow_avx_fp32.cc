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
#ifdef ENABLE_AVX
#include "src/litert/kernel/cpu/fp32/convolution_slidewindow_avx_fp32.h"
#include "nnacl/fp32/conv_common_fp32.h"
#include "nnacl/fp32/conv_1x1_x86_fp32.h"

namespace mindspore::kernel {
void ConvolutionSWAVXCPUKernel::InitGlobalVariable() {
  oc_tile_ = C8NUM;
  oc_res_ = conv_param_->output_channel_ % oc_tile_;
  if (conv_param_->kernel_h_ == 1 && conv_param_->kernel_w_ == 1) {
    // 1x1 conv is aligned to C8NUM
    in_tile_ = C8NUM;
    ic_res_ = conv_param_->input_channel_ % in_tile_;
  }
}

int ConvolutionSWAVXCPUKernel::RunImpl(int task_id) {
  if (conv_param_->kernel_w_ == 1 && conv_param_->kernel_h_ == 1) {
    Conv1x1SWAVXFp32(input_data_, reinterpret_cast<float *>(packed_weight_), reinterpret_cast<float *>(bias_data_),
                     output_data_, task_id, conv_param_, slidingWindow_param_);
  } else {
    ConvSWAVXFp32(input_data_, reinterpret_cast<float *>(packed_weight_), reinterpret_cast<float *>(bias_data_),
                  output_data_, task_id, conv_param_, slidingWindow_param_);
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
#endif  // ENABLE_AVX
