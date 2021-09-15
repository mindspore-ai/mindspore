/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp16/group_convolution_fp16.h"
#include "src/runtime/kernel/arm/fp16/convolution_delegate_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionFP16CPUKernel::SeparateInput(int group_id) {
  // input may either be float32 or float16
  auto in_tensor = in_tensors_.front();
  int in_plane = in_tensor->Height() * in_tensor->Width() * in_tensor->Batch();
  int sub_in_channel = conv_param_->input_channel_;
  int ori_in_channel = sub_in_channel * group_num_;
  auto sub_in_data = static_cast<lite::Tensor *>(group_convs_.at(group_id)->in_tensors().front())->data();
  MS_ASSERT(sub_in_data != nullptr);
  auto in_data_type = in_tensors_.front()->data_type();
  auto sub_in_data_type = group_convs_.at(group_id)->in_tensors().front()->data_type();
  if (in_data_type != sub_in_data_type) {
    MS_LOG(ERROR) << "data type of sub conv kernel input should be the same as origin input's.";
    return RET_ERROR;
  }
  if (!(in_data_type == kNumberTypeFloat32 || in_data_type == kNumberTypeFloat16)) {
    MS_LOG(ERROR) << "Invalid data type.";
    return RET_ERROR;
  }
  if (in_tensors_.front()->data_type() == kNumberTypeFloat16) {
    float16_t *src_ptr = reinterpret_cast<float16_t *>(ori_in_data_) + group_id * sub_in_channel;
    float16_t *dst_ptr = reinterpret_cast<float16_t *>(sub_in_data);
    MS_ASSERT(src_ptr);
    MS_ASSERT(dst_ptr);
    for (int i = 0; i < in_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel * sizeof(float16_t));
      src_ptr += ori_in_channel;
      dst_ptr += sub_in_channel;
    }
  } else {
    float *src_ptr = reinterpret_cast<float *>(ori_in_data_) + group_id * sub_in_channel;
    float *dst_ptr = reinterpret_cast<float *>(sub_in_data);
    MS_ASSERT(src_ptr);
    MS_ASSERT(dst_ptr);
    for (int i = 0; i < in_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel * sizeof(float));
      src_ptr += ori_in_channel;
      dst_ptr += sub_in_channel;
    }
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::PostConcat(int group_id) {
  // output is must float16 data type
  auto out_tensor = out_tensors_.front();
  int out_plane = out_tensor->Height() * out_tensor->Width() * out_tensor->Batch();
  int sub_out_channel = conv_param_->output_channel_;
  int ori_out_channel = sub_out_channel * group_num_;
  auto sub_out_data = reinterpret_cast<float16_t *>(
    static_cast<lite::Tensor *>(group_convs_.at(group_id)->out_tensors().front())->data());
  MS_ASSERT(sub_out_data);
  float16_t *src_ptr = sub_out_data;
  float16_t *dst_ptr = reinterpret_cast<float16_t *>(ori_out_data_) + group_id * sub_out_channel;
  for (int i = 0; i < out_plane; ++i) {
    memcpy(dst_ptr, src_ptr, sub_out_channel * sizeof(float16_t));
    src_ptr += sub_out_channel;
    dst_ptr += ori_out_channel;
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (group_conv_creator_ == nullptr) {
    return lite::RET_ERROR;
  }
  group_conv_creator_->SetShapeOfTensors();
  for (int i = 0; i < conv_param_->group_; ++i) {
    auto *new_conv_param = CreateNewConvParameter(conv_param_);
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    auto ret = group_conv_creator_->GetSingleConvParam(new_conv_param, &new_inputs, &new_outputs, i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetSingleConv for fp16 group conv failed.";
      return lite::RET_ERROR;
    }
    group_convs_.emplace_back(new (std::nothrow) ConvolutionDelegateFP16CPUKernel(
      reinterpret_cast<OpParameter *>(new_conv_param), new_inputs, new_outputs, ctx_));
  }
  return GroupConvolutionBaseCPUKernel::Init();
}
}  // namespace mindspore::kernel
