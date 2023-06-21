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
#include "src/litert/kernel/cpu/fp32/convolution_slidewindow_arm64_fp32.h"
#include "nnacl/fp32/conv_sw_arm64_fp32.h"

namespace mindspore::kernel {
void ConvolutionSWARM64CPUKernel::InitGlobalVariable() {
  oc_tile_ = C8NUM;
  oc_res_ = conv_param_->output_channel_ % oc_tile_;
}

int ConvolutionSWARM64CPUKernel::RunImpl(int task_id) {
  ConvSWArm64Fp32(input_data_, reinterpret_cast<float *>(packed_weight_), reinterpret_cast<float *>(bias_data_),
                  output_data_, task_id, conv_param_, slidingWindow_param_);
  return RET_OK;
}
}  // namespace mindspore::kernel
