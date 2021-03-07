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
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PReLUFusion;

namespace mindspore::kernel {
namespace {
int PReluRun(void *cdata, int task_id) {
  auto PRelu = reinterpret_cast<PReluCPUKernel *>(cdata);
  auto ret = PRelu->DoExcute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PReluRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace

int PReluCPUKernel::Init() {
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
  if (prelu_param_->channelShared) {
    PReluShareChannel(input_data_, output_data_, prelu_param_, task_id);
  } else {
    int res_plane = prelu_param_->input_num_ - task_id * prelu_param_->tile_block_;
    int plane = MSMIN(prelu_param_->tile_block_, res_plane);
    if (plane <= 0) {
      return RET_OK;
    }
    float *in = input_data_ + task_id * prelu_param_->tile_block_ * prelu_param_->channel_num_;
    float *out = output_data_ + task_id * prelu_param_->tile_block_ * prelu_param_->channel_num_;
    PRelu(in, out, prelu_param_, plane);
  }
  return RET_OK;
}

int PReluCPUKernel::ReSize() {
  if (prelu_param_->channelShared) {
    return RET_OK;
  }

  auto input_tensor = in_tensors_.at(0);
  auto in_shape = input_tensor->shape();
  auto n_dim = in_shape.size();
  auto channel_num = in_shape.at(n_dim - 1);
  int input_plane = 1;
  for (size_t i = 0; i < n_dim - 1; ++i) {
    input_plane *= in_shape.at(i);
  }

  prelu_param_->input_num_ = input_plane;
  prelu_param_->tile_block_ = UP_DIV(UP_DIV(input_plane, TILE_NUM), op_parameter_->thread_num_) * TILE_NUM;
  prelu_param_->channel_num_ = channel_num;
  return RET_OK;
}

int PReluCPUKernel::ProcessShareChannelInput() {
  auto input_tensor = in_tensors_.at(0);
  prelu_param_->input_num_ = input_tensor->ElementsNum();
  int tile = 32;
#if defined(ENABLE_ARM64) || defined(ENABLE_AVX)
  tile = 64;
#endif
  prelu_param_->tile_block_ = UP_DIV(prelu_param_->input_num_, tile);
  input_data_ =
    reinterpret_cast<float *>(context_->allocator->Malloc(prelu_param_->tile_block_ * tile * sizeof(float)));
  if (input_data_ == nullptr) {
    MS_LOG(ERROR) << "malloc input_data_ failed.";
    return RET_ERROR;
  }
  memcpy(input_data_, ori_input_, prelu_param_->input_num_ * sizeof(float));
  return RET_OK;
}

int PReluCPUKernel::Run() {
  MS_ASSERT(in_tensors_.size() >= 2);
  auto input_tensor = in_tensors_[0];
  ori_input_ = reinterpret_cast<float *>(input_tensor->data_c());
  output_data_ = reinterpret_cast<float *>(out_tensors_.at(kOutputIndex)->data_c());
  MS_ASSERT(ori_input_);
  MS_ASSERT(output_data_);
  if (prelu_param_->channelShared) {
    auto ret = ProcessShareChannelInput();
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "ProcessShareChannel failed.";
      return ret;
    }
  } else {
    input_data_ = ori_input_;
  }

  // negative slope tensor
  auto negative_slope_tensor = in_tensors_.at(1);
  prelu_param_->slope_ = reinterpret_cast<float *>(negative_slope_tensor->data_c());

  auto ret = ParallelLaunch(this->context_->thread_pool_, PReluRun, this, prelu_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "PRelu Run error: error_code[" << ret << "]";
    context_->allocator->Free(input_data_);
    return RET_ERROR;
  }

  if (prelu_param_->channelShared) {
    memcpy(output_data_, input_data_, prelu_param_->input_num_ * sizeof(float));
    context_->allocator->Free(input_data_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PReLUFusion, LiteKernelCreator<PReluCPUKernel>)
}  // namespace mindspore::kernel
