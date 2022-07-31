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

#include "src/litert/kernel/cpu/fp16/group_convolution_fp16.h"
#include "src/litert/kernel/cpu/fp16/convolution_delegate_fp16.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionFP16CPUKernel::Separate(int task_id) {
  auto plane_step = UP_DIV(in_plane_, in_thread_num_);
  auto begin_plane = plane_step * task_id;
  auto end_plane = MSMIN(in_plane_, plane_step * (task_id + 1));

  if (in_data_type_ == kNumberTypeFloat16) {
    auto src_ptr = reinterpret_cast<float16_t *>(sub_in_src_) + begin_plane * ori_in_channel_;
    auto dst_ptr = reinterpret_cast<float16_t *>(sub_in_dst_) + begin_plane * sub_in_channel_;
    for (int i = begin_plane; i < end_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel_ * sizeof(float16_t));
      src_ptr += ori_in_channel_;
      dst_ptr += sub_in_channel_;
    }
  } else if (in_data_type_ == kNumberTypeFloat32) {
    auto src_ptr = reinterpret_cast<float *>(sub_in_src_) + begin_plane * ori_in_channel_;
    auto dst_ptr = reinterpret_cast<float *>(sub_in_dst_) + begin_plane * sub_in_channel_;
    for (int i = begin_plane; i < end_plane; ++i) {
      memcpy(dst_ptr, src_ptr, sub_in_channel_ * sizeof(float));
      src_ptr += ori_in_channel_;
      dst_ptr += sub_in_channel_;
    }
  } else {
    MS_LOG(ERROR) << "Invalid data type.";
    return RET_ERROR;
  }
  return RET_OK;
}

int SeparateInputFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<GroupConvolutionFP16CPUKernel *>(cdata);
  auto ret = kernel->Separate(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution fp16 separate input error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::SeparateInput(int group_id) {
  // input may either be float32 or float16
  auto sub_in_data = group_convs_.at(group_id)->in_tensors().front()->data();
  CHECK_NULL_RETURN(sub_in_data);
  in_data_type_ = in_tensors_.front()->data_type();
  auto sub_in_data_type = group_convs_.at(group_id)->in_tensors().front()->data_type();
  if (in_data_type_ != sub_in_data_type) {
    MS_LOG(ERROR) << "data type of sub conv kernel input should be the same as origin input's.";
    return RET_ERROR;
  }

  sub_in_dst_ = sub_in_data;
  CHECK_NULL_RETURN(sub_in_dst_);
  if (in_data_type_ == kNumberTypeFloat16) {
    sub_in_src_ = reinterpret_cast<float16_t *>(ori_in_data_) + group_id * sub_in_channel_;
    CHECK_NULL_RETURN(sub_in_src_);
  } else if (in_data_type_ == kNumberTypeFloat32) {
    sub_in_src_ = reinterpret_cast<float *>(ori_in_data_) + group_id * sub_in_channel_;
    CHECK_NULL_RETURN(sub_in_src_);
  } else {
    MS_LOG(ERROR) << "Invalid data type.";
    return RET_ERROR;
  }

  auto ret = ParallelLaunch(this->ms_context_, SeparateInputFp16Run, this, in_thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution fp16 separate input error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::Concat(int task_id) {
  auto plane_step = UP_DIV(out_plane_, out_thread_num_);
  auto begin_plane = plane_step * task_id;
  auto end_plane = MSMIN(out_plane_, plane_step * (task_id + 1));
  auto src_ptr = sub_out_src_ + begin_plane * sub_out_channel_;
  auto dst_ptr = sub_out_dst_ + begin_plane * ori_out_channel_;
  for (int i = begin_plane; i < end_plane; ++i) {
    memcpy(dst_ptr, src_ptr, sub_out_channel_ * sizeof(float16_t));
    src_ptr += sub_out_channel_;
    dst_ptr += ori_out_channel_;
  }
  return RET_OK;
}

int ConcatOutputFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto kernel = reinterpret_cast<GroupConvolutionFP16CPUKernel *>(cdata);
  auto ret = kernel->Concat(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution fp16 concat output error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::PostConcat(int group_id) {
  // output is must float16 data type
  sub_out_src_ = reinterpret_cast<float16_t *>(group_convs_.at(group_id)->out_tensors().front()->data());
  sub_out_dst_ = reinterpret_cast<float16_t *>(ori_out_data_) + group_id * sub_out_channel_;
  CHECK_NULL_RETURN(sub_out_src_);
  CHECK_NULL_RETURN(sub_out_dst_);

  auto ret = ParallelLaunch(this->ms_context_, ConcatOutputFp16Run, this, out_thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution fp16 concat output error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionFP16CPUKernel::Prepare() {
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
    auto kernel = new (std::nothrow)
      ConvolutionDelegateFP16CPUKernel(reinterpret_cast<OpParameter *>(new_conv_param), new_inputs, new_outputs, ctx_);
    if (kernel == nullptr) {
      MS_LOG(ERROR) << "Create kernel failed.";
      return lite::RET_ERROR;
    }
    group_convs_.push_back(kernel);
  }
  return GroupConvolutionBaseCPUKernel::Prepare();
}
}  // namespace mindspore::kernel
