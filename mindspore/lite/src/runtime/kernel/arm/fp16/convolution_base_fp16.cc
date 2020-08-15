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
#include "src/runtime/kernel/arm/nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_factory.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

namespace mindspore::kernel {
int ConvolutionBaseFP16CPUKernel::GetExecuteTensor() {
  // ===================input====================//
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto input_data_type = input_tensor->data_type();
  MS_ASSERT(input_data_type == kNumberTypeFloat32 || input_data_type == kNumberTypeFloat16);
  if (input_data_type == kNumberTypeFloat32) {
    auto input_ele_num = input_tensor->ElementsNum();
    auto ori_input_data = reinterpret_cast<float *>(input_tensor->Data());
    Float32ToFloat16(ori_input_data, fp16_input_, input_ele_num);
    execute_input_ = fp16_input_;
  } else {
    auto ori_input_data = reinterpret_cast<float16_t *>(input_tensor->Data());
    execute_input_ = ori_input_data;
  }
  // ==================output====================//
  auto out_tensor = out_tensors_.at(kOutputIndex);
  auto out_data_type = out_tensor->data_type();
  MS_ASSERT(out_data_type == kNumberTypeFloat32 || out_data_type == kNumberTypeFloat16);
  out_data_type_ = out_data_type;
  if (out_data_type == kNumberTypeFloat32) {
    execute_output_ = fp16_out_;
  } else {
    auto out_ptr = reinterpret_cast<float16_t *>(out_tensor->Data());
    execute_output_ = out_ptr;
  }
  return RET_OK;
}

int ConvolutionBaseFP16CPUKernel::GetExecuteFilter() {
  auto weight_tensor = in_tensors_.at(kWeightIndex);
  auto weight_data_type = weight_tensor->data_type();
  MS_ASSERT(weight_data_type == kNumberTypeFloat32 || weight_data_type == kNumberTypeFloat16);
  if (weight_data_type == kNumberTypeFloat32) {
    float *origin_weight = reinterpret_cast<float *>(in_tensors_.at(kWeightIndex)->Data());
    size_t fp16_weight_size = conv_param_->input_channel_ * conv_param_->output_channel_ * conv_param_->kernel_h_ *
                              conv_param_->input_w_ * sizeof(float16_t);
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
    Float16ToFloat32(fp16_out_, output_addr, out_ele_num);
  }
}

}  // namespace mindspore::kernel
