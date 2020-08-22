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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
int ConvolutionBaseFP16CPUKernel::GetExecuteTensor() {
  // ===================input====================//
  auto input_tensor = in_tensors_.at(kInputIndex);
  in_data_type_ = input_tensor->data_type();
  MS_ASSERT(in_data_type_ == kNumberTypeFloat32 || in_data_type_ == kNumberTypeFloat16);

  execute_input_ = ConvertInputFp32toFp16(input_tensor, context_);

  // ==================output====================//
  auto out_tensor = out_tensors_.at(kOutputIndex);
  out_data_type_ = out_tensor->data_type();
  MS_ASSERT(out_data_type_ == kNumberTypeFloat32 || out_data_type_ == kNumberTypeFloat16);

  execute_output_ = MallocOutputFp16(out_tensor, context_);
  return RET_OK;
}

int ConvolutionBaseFP16CPUKernel::GetExecuteFilter() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_data_type = weight_tensor->data_type();

  auto input_channel = weight_tensor->Channel();
  auto output_channel = weight_tensor->Batch();
  auto kernel_h = weight_tensor->Height();
  auto kernel_w = weight_tensor->Width();

  MS_ASSERT(weight_data_type == kNumberTypeFloat32 || weight_data_type == kNumberTypeFloat16);
  if (weight_data_type == kNumberTypeFloat32) {
    float *origin_weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->Data());
    size_t fp16_weight_size = input_channel * output_channel * kernel_h * kernel_w * sizeof(float16_t);
    fp16_weight_ = reinterpret_cast<float16_t *>(malloc(fp16_weight_size));
    if (fp16_weight_ == nullptr) {
      MS_LOG(ERROR) << "malloc fp16_weight_ failed.";
      return RET_ERROR;
    }
    for (int i = 0; i < fp16_weight_size / sizeof(float16_t); ++i) {
      fp16_weight_[i] = (float16_t)origin_weight[i];
    }
    execute_weight_ = fp16_weight_;
  } else {
    auto *origin_weight = reinterpret_cast<float16_t *>(in_tensors_.at(kWeightIndex)->Data());
    execute_weight_ = origin_weight;
  }
  return RET_OK;
}

void ConvolutionBaseFP16CPUKernel::IfCastOutput() {
  if (out_data_type_ == kNumberTypeFloat32) {
    auto out_tensor = out_tensors_.at(kOutputIndex);
    auto out_ele_num = out_tensor->ElementsNum();
    auto output_addr = reinterpret_cast<float *>(out_tensor->Data());
    Float16ToFloat32(execute_output_, output_addr, out_ele_num);
  }
}

void ConvolutionBaseFP16CPUKernel::FreeTmpBuffer() {
  if (in_data_type_ == kNumberTypeFloat32) {
    context_->allocator->Free(execute_input_);
  }
  if (out_data_type_ == kNumberTypeFloat32) {
    context_->allocator->Free(execute_output_);
  }
}

}  // namespace mindspore::kernel
