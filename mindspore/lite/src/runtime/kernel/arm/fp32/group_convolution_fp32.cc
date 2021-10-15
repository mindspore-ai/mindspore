/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp32/group_convolution_fp32.h"
#include "src/runtime/kernel/arm/fp32/convolution_delegate_fp32.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionFp32CPUKernel::SeparateInput(int group_id) {
  auto in_tensor = in_tensors_.front();
  int in_plane = in_tensor->Height() * in_tensor->Width() * in_tensor->Batch();
  if (in_plane < 0) {
    MS_LOG(ERROR) << "get in_plane from in_tensor failed.";
    return RET_ERROR;
  }
  int sub_in_channel = conv_param_->input_channel_;
  int ori_in_channel = sub_in_channel * group_num_;
  auto sub_in_data =
    reinterpret_cast<float *>(static_cast<lite::Tensor *>(group_convs_.at(group_id)->in_tensors().front())->data());
  float *src_ptr = reinterpret_cast<float *>(ori_in_data_) + group_id * sub_in_channel;
  float *dst_ptr = sub_in_data;
  for (int i = 0; i < in_plane; ++i) {
    memcpy(dst_ptr, src_ptr, sub_in_channel * sizeof(float));
    src_ptr += ori_in_channel;
    dst_ptr += sub_in_channel;
  }
  return RET_OK;
}

int GroupConvolutionFp32CPUKernel::PostConcat(int group_id) {
  auto out_tensor = out_tensors_.front();
  int out_plane = out_tensor->Height() * out_tensor->Width() * out_tensor->Batch();
  if (out_plane < 0) {
    MS_LOG(ERROR) << "get out_plane from out_tensor failed.";
    return RET_ERROR;
  }
  int sub_out_channel = conv_param_->output_channel_;
  int ori_out_channel = sub_out_channel * group_num_;
  auto sub_out_data =
    reinterpret_cast<float *>(static_cast<lite::Tensor *>(group_convs_.at(group_id)->out_tensors().front())->data());
  float *src_ptr = sub_out_data;
  float *dst_ptr = reinterpret_cast<float *>(ori_out_data_) + group_id * sub_out_channel;
  for (int i = 0; i < out_plane; ++i) {
    memcpy(dst_ptr, src_ptr, sub_out_channel * sizeof(float));
    src_ptr += sub_out_channel;
    dst_ptr += ori_out_channel;
  }
  return RET_OK;
}

int GroupConvolutionFp32CPUKernel::Init() {
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
      MS_LOG(ERROR) << "GetSingleConv for fp32 group conv failed.";
      return lite::RET_ERROR;
    }
    auto new_conv = new (std::nothrow)
      ConvolutionDelegateCPUKernel(reinterpret_cast<OpParameter *>(new_conv_param), new_inputs, new_outputs, ctx_);
    if (new_conv == nullptr) {
      MS_LOG(ERROR) << "malloc new conv error.";
      return lite::RET_ERROR;
    }
    (void)group_convs_.emplace_back(new_conv);
  }
  return GroupConvolutionBaseCPUKernel::Init();
}
}  // namespace mindspore::kernel
