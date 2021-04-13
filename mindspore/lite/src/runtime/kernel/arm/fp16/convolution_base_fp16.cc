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

#include "src/runtime/kernel/arm/fp16/convolution_base_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
ConvolutionBaseFP16CPUKernel::~ConvolutionBaseFP16CPUKernel() {
  if (fp16_weight_ != nullptr) {
    free(fp16_weight_);
    fp16_weight_ = nullptr;
  }
}

int ConvolutionBaseFP16CPUKernel::GetExecuteTensor() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);
  execute_input_ = reinterpret_cast<float16_t *>(input_tensor->data_c());
  execute_output_ = reinterpret_cast<float16_t *>(output_tensor->data_c());
  return RET_OK;
}

int ConvolutionBaseFP16CPUKernel::GetExecuteFilter(lite::Tensor *weight_tensor, void *origin_data) {
  MS_ASSERT(origin_weight_data_type_ == kNumberTypeFloat32 || origin_weight_data_type_ == kNumberTypeFloat16);
  if (origin_weight_data_type_ == kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Conv fp16 only support fp16 weight";
    return RET_ERROR;
  } else {
    execute_weight_ = reinterpret_cast<float16_t *>(origin_data);
    fp16_weight_ = nullptr;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
