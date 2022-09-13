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

#include "src/litert/kernel/cpu/int8/group_convolution_int8.h"
#include "src/litert/kernel/cpu/int8/convolution_int8_creator.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int GroupConvolutionInt8CPUKernel::Separate(int task_id) {
  auto plane_step = UP_DIV(in_plane_, in_thread_num_);
  auto begin_plane = plane_step * task_id;
  auto end_plane = MSMIN(in_plane_, plane_step * (task_id + 1));
  auto src_ptr = sub_in_src_ + begin_plane * ori_in_channel_;
  auto dst_ptr = sub_in_dst_ + begin_plane * sub_in_channel_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, static_cast<size_t>(sub_in_channel_));
    src_ptr += ori_in_channel_;
    dst_ptr += sub_in_channel_;
  }
  return RET_OK;
}

int SeparateInputInt8Run(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<GroupConvolutionInt8CPUKernel *>(cdata);
  auto ret = kernel->Separate(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution separate input error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionInt8CPUKernel::SeparateInput(int group_id) {
  sub_in_src_ = reinterpret_cast<int8_t *>(ori_in_data_) + group_id * sub_in_channel_;
  sub_in_dst_ = reinterpret_cast<int8_t *>(group_convs_.at(group_id)->in_tensors().front()->data());
  CHECK_NULL_RETURN(sub_in_src_);
  CHECK_NULL_RETURN(sub_in_dst_);

  auto ret = ParallelLaunch(this->ms_context_, SeparateInputInt8Run, this, in_thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution separate input error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionInt8CPUKernel::Concat(int task_id) {
  auto plane_step = UP_DIV(out_plane_, out_thread_num_);
  auto begin_plane = plane_step * task_id;
  auto end_plane = MSMIN(out_plane_, plane_step * (task_id + 1));
  auto src_ptr = sub_out_src_ + begin_plane * sub_out_channel_;
  auto dst_ptr = sub_out_dst_ + begin_plane * ori_out_channel_;
  for (int i = begin_plane; i < end_plane; ++i) {
    (void)memcpy(dst_ptr, src_ptr, static_cast<size_t>(sub_out_channel_));
    src_ptr += sub_out_channel_;
    dst_ptr += ori_out_channel_;
  }
  return RET_OK;
}

int ConcatOutputInt8Run(void *cdata, int task_id, float, float) {
  auto kernel = reinterpret_cast<GroupConvolutionInt8CPUKernel *>(cdata);
  auto ret = kernel->Concat(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution concat output error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionInt8CPUKernel::PostConcat(int group_id) {
  sub_out_src_ = reinterpret_cast<int8_t *>(group_convs_.at(group_id)->out_tensors().front()->data());
  sub_out_dst_ = reinterpret_cast<int8_t *>(ori_out_data_) + group_id * sub_out_channel_;
  CHECK_NULL_RETURN(sub_out_src_);
  CHECK_NULL_RETURN(sub_out_dst_);

  auto ret = ParallelLaunch(this->ms_context_, ConcatOutputInt8Run, this, out_thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Group convolution concat output error";
    return RET_ERROR;
  }
  return RET_OK;
}

int GroupConvolutionInt8CPUKernel::Prepare() {
  if (group_conv_creator_ == nullptr) {
    return lite::RET_ERROR;
  }
  auto ret = group_conv_creator_->SetShapeOfTensors();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SetShapeOfTensors for int8 group conv failed.";
    return lite::RET_ERROR;
  }
  for (int i = 0; i < conv_param_->group_; ++i) {
    auto *new_conv_param = CreateNewConvParameter(conv_param_);
    std::vector<lite::Tensor *> new_inputs;
    std::vector<lite::Tensor *> new_outputs;
    ret = group_conv_creator_->GetSingleConvParam(new_conv_param, &new_inputs, &new_outputs, i);
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "GetSingleConv for int8 group conv failed.";
      return lite::RET_ERROR;
    }
    group_conv_creator_->CopyQuantParam(&new_inputs);
    (void)group_convs_.emplace_back(
      CpuConvInt8KernelSelect(new_inputs, new_outputs, reinterpret_cast<OpParameter *>(new_conv_param), ctx_));
  }
  return GroupConvolutionBaseCPUKernel::Prepare();
}
}  // namespace mindspore::kernel
