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
#include "src/litert/kernel/cpu/fp32/prelu_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/prelu_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::kernel {
static int PReluRun(void *cdata, int task_id, float, float) {
  auto PRelu = reinterpret_cast<const PReluCPUKernel *>(cdata);
  auto ret = PRelu->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PReluCPUKernel::Prepare() {
  constexpr int kSlopeIndex = 1;
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kInputIndex]);
  CHECK_NULL_RETURN(in_tensors_[kSlopeIndex]);
  CHECK_NULL_RETURN(out_tensors_[kOutputIndex]);
  auto slope_shapes = in_tensors_[C1NUM]->ElementsNum();
  auto input_channel = in_tensors_[C0NUM]->Channel();
  if ((slope_shapes != C1NUM) && (slope_shapes != input_channel)) {
    MS_LOG(ERROR) << "slope_shapes: " << slope_shapes << " is not equal to 1 or input_channel: " << input_channel;
    return lite::RET_ERROR;
  }
  if (in_tensors_[1]->ElementsNum() == 1) {
    param_->channelShared = true;
  } else {
    param_->channelShared = false;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PReluCPUKernel::DoExcute(int task_id) const {
  int thread_num = param_->op_parameter_.thread_num_;
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread_num is 0!";
    return RET_ERROR;
  }
  int num = param_->channelShared ? param_->input_num_ : param_->input_num_ / param_->channel_num_;
  int step = UP_DIV(num, thread_num);
  int start = task_id * step;
  int end = MSMIN(start + step, num);

  if (param_->channelShared) {
    PReluShareChannel(static_cast<float *>(input_data_), static_cast<float *>(output_data_),
                      static_cast<float *>(slope_data_)[0], start, end);
  } else {
    PRelu(static_cast<float *>(input_data_), static_cast<float *>(output_data_), static_cast<float *>(slope_data_),
          start, end, param_->channel_num_);
  }
  return RET_OK;
}

int PReluCPUKernel::ReSize() {
  auto &input = in_tensors_[kInputIndex];
  param_->input_num_ = input->ElementsNum();
  CHECK_NOT_EQUAL_RETURN(out_tensors_.front()->ElementsNum(), param_->input_num_);
  if (input->Channel() == RET_ERROR) {
    MS_LOG(ERROR) << "get channel failed.";
    return RET_ERROR;
  }
  param_->channel_num_ = input->Channel();
  return RET_OK;
}

int PReluCPUKernel::Run() {
  const int kSlopeIndex = 1;
  input_data_ = in_tensors_[kInputIndex]->data();
  slope_data_ = in_tensors_[kSlopeIndex]->data();
  output_data_ = out_tensors_[kOutputIndex]->data();
  CHECK_NULL_RETURN(input_data_);
  CHECK_NULL_RETURN(slope_data_);
  CHECK_NULL_RETURN(output_data_);

  auto ret = ParallelLaunch(this->ms_context_, PReluRun, this, param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PRelu Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PReLUFusion, LiteKernelCreator<PReluCPUKernel>)
}  // namespace mindspore::kernel
