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
#include "src/runtime/kernel/arm/fp32/prelu_fp32.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::kernel {
static int PReluRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto PRelu = reinterpret_cast<PReluCPUKernel *>(cdata);
  auto ret = PRelu->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PReluCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), C2NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (in_tensors_[1]->ElementsNum() == 1) {
    prelu_param_->channelShared = true;
  } else {
    prelu_param_->channelShared = false;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int PReluCPUKernel::DoExcute(int task_id) {
  int thread_num = prelu_param_->op_parameter_.thread_num_;
  if (thread_num == 0) {
    MS_LOG(ERROR) << "thread_num is 0!";
    return RET_ERROR;
  }
  if (prelu_param_->channelShared) {
    int step = UP_DIV(prelu_param_->input_num_, thread_num);
    int start = task_id * step;
    int end = MSMIN(start + step, prelu_param_->input_num_);
    PReluShareChannel(input_data_, output_data_, prelu_param_->slope_[0], start, end);
  } else {
    int step = UP_DIV(prelu_param_->tile_block_, thread_num);
    int start = task_id * step;
    int end = MSMIN(start + step, prelu_param_->tile_block_);
    PRelu(input_data_, output_data_, prelu_param_->slope_, start, end, prelu_param_->channel_num_);
  }
  return RET_OK;
}

int PReluCPUKernel::ReSize() {
  auto input_tensor = in_tensors_.at(0);
  auto in_shape = input_tensor->shape();
  auto n_dim = in_shape.size();
  auto channel_num = in_shape.at(n_dim - 1);
  int input_plane = 1;
  for (size_t i = 0; i < n_dim - 1; ++i) {
    input_plane *= in_shape.at(i);
  }
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(input_plane, channel_num), RET_ERROR);
  prelu_param_->input_num_ = input_plane * channel_num;
  prelu_param_->tile_block_ = input_plane;
  prelu_param_->channel_num_ = channel_num;
  return RET_OK;
}

int PReluCPUKernel::Run() {
  auto input_tensor = in_tensors_[0];
  input_data_ = reinterpret_cast<float *>(input_tensor->data());
  output_data_ = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data());
  CHECK_NULL_RETURN(input_data_);
  CHECK_NULL_RETURN(output_data_);

  // negative slope tensor
  auto negative_slope_tensor = in_tensors_.at(1);
  CHECK_NULL_RETURN(negative_slope_tensor->data());
  prelu_param_->slope_ = reinterpret_cast<float *>(negative_slope_tensor->data());

  auto ret = ParallelLaunch(this->ms_context_, PReluRun, this, prelu_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PRelu Run error: error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PReLUFusion, LiteKernelCreator<PReluCPUKernel>)
}  // namespace mindspore::kernel
