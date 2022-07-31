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

#include "src/litert/kernel/cpu/fp16_grad/bn_fp16_grad.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp16_grad/batch_norm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNormGrad;

namespace mindspore::kernel {
namespace {
constexpr int kNumInputDim_0 = 0;
constexpr int kNumInputDim_1 = 1;
constexpr int kNumInputDim_2 = 2;
constexpr int kNumInputDim_3 = 3;
constexpr int kNumInputDim_4 = 4;
constexpr int kNumInputDim_5 = 4;
constexpr int kNumOutputDim_2 = 2;
constexpr int kNumJobs = 4;
}  // namespace
int BNGradCPUKernelFp16::ReSize() {
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), 5);
  CHECK_LESS_RETURN(out_tensors_.size(), 3);
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_0));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_1));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_2));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_3));
  CHECK_NULL_RETURN(in_tensors_.at(kNumInputDim_4));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(kNumOutputDim_2));
  auto *input_x = in_tensors_.at(1);
  int channels = input_x->shape().at(kNHWC_C);
  ws_size_ = 2 * channels;
  set_workspace_size(ws_size_ * sizeof(float16_t));
  return RET_OK;
}

int BNGradCPUKernelFp16::Prepare() {
  for (int i = 0; i < in_tensors_.size(); i++) {
    if (in_tensors_.at(i)->data_type() != kNumberTypeFloat16) {
      MS_LOG(ERROR) << "BNGradCPUKernelFp16 type error in_tensor_[" << i << "]";
    }
  }
  return ReSize();
}

int BNGradCPUKernelFp16::DoExecute(int task_id) {
  auto *input_yt = in_tensors_.at(kNumInputDim_0);
  auto *input_x = in_tensors_.at(kNumInputDim_1);
  auto *input_scale = in_tensors_.at(kNumInputDim_2);
  auto *input_mean = in_tensors_.at(kNumInputDim_3);
  auto *input_var = in_tensors_.at(kNumInputDim_4);

  auto kernel_name = this->name();
  if (kernel_name.find("FusedBatchNormGradCPU") != std::string::npos) {
    input_mean = in_tensors_.at(kNumInputDim_4);
    input_var = in_tensors_.at(kNumInputDim_5);
  }
  auto bn_param = reinterpret_cast<BNGradParameter *>(op_parameter_);
  int stage = stage_;
  int thread_num = thread_num_;
  float16_t *save_mean = reinterpret_cast<float16_t *>(input_mean->data());
  CHECK_NULL_RETURN(save_mean);
  float16_t *save_var = reinterpret_cast<float16_t *>(input_var->data());
  CHECK_NULL_RETURN(save_var);

  auto *output_dx = out_tensors_.at(0);
  auto *output_scale = out_tensors_.at(1);
  auto *output_bias = out_tensors_.at(kNumOutputDim_2);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();

  float *workspace_temp = static_cast<float *>(workspace());
  CHECK_NULL_RETURN(workspace_temp);
  float *dxhat_sum = workspace_temp;
  float *dxhathat_sum = dxhat_sum + channels;
  float16_t *x = reinterpret_cast<float16_t *>(input_x->data());
  float16_t *yt = reinterpret_cast<float16_t *>(input_yt->data());
  float16_t *scale = reinterpret_cast<float16_t *>(input_scale->data());
  float16_t *dx = reinterpret_cast<float16_t *>(output_dx->data());
  float16_t *dbias = reinterpret_cast<float16_t *>(output_bias->data());
  float16_t *dscale = reinterpret_cast<float16_t *>(output_scale->data());
  int total = spatial * batch;
  int stride = UP_DIV(total, thread_num);
  int count = MSMIN(stride, total - stride * task_id);
  count = (count < 0) ? 0 : count;
  switch (stage) {
    case 0: {
      for (int job = task_id; job < kNumJobs; job += thread_num) {
        switch (job) {
          case 0:
            var2InvarFp16(save_var, input_var->ElementsNum(), bn_param->epsilon_);
            break;
          case 1:
            std::fill(workspace_temp, workspace_temp + ws_size_, 0.f);
            break;
          case 2:
            std::fill(dbias, dbias + channels, 0.f);
            break;
          case 3:
            std::fill(dscale, dscale + channels, 0.f);
            break;
        }
      }
      if (thread_num == 1) {
        backwardAllFp16(x, yt, save_mean, save_var, scale, total, channels, dxhat_sum, dxhathat_sum, dbias, dscale, dx);
      }
      break;
    }
    case 1: {
      backwardP1Fp16(x, yt, save_mean, save_var, scale, total, channels, dxhat_sum, dxhathat_sum, dbias, dscale);
      break;
    }
    case 2: {
      backwardP2Fp16(x + task_id * stride * channels, yt + task_id * stride * channels, save_mean, save_var, scale,
                     count, total, channels, dxhat_sum, dxhathat_sum, dx + task_id * stride * channels);
      break;
    }
  }

  return RET_OK;
}

int BNGradFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernelFp16 *>(cdata);
  auto error_code = bn_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernelFp16::Run() {
  stage_ = 0;
  thread_num_ = ms_context_->thread_num_;
  if (thread_num_ == 1) {
    int error_code = ParallelLaunch(this->ms_context_, BNGradFp16Run, this, thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    const std::vector<int> threads = {thread_num_, 1, thread_num_};
    for (size_t stage = 0; stage < threads.size(); stage++) {
      stage_ = static_cast<int>(stage);
      int error_code = ParallelLaunch(this->ms_context_, BNGradFp16Run, this, threads.at(stage));
      if (error_code != RET_OK) {
        MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_BatchNormGrad, LiteKernelCreator<BNGradCPUKernelFp16>)
}  // namespace mindspore::kernel
