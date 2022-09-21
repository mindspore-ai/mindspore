/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32_grad/bn_grad.h"
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNormGrad;

namespace mindspore::kernel {
namespace {
constexpr int kMaxTaskNum = 3;
constexpr int kNumInputDim2 = 2;
constexpr int kNumInputDim4 = 4;
}  // namespace

int BNGradCPUKernel::ReSize() { return RET_OK; }

int BNGradCPUKernel::Prepare() {
  CHECK_NULL_RETURN(op_parameter_);
  CHECK_LESS_RETURN(in_tensors_.size(), 6);
  CHECK_LESS_RETURN(out_tensors_.size(), 3);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[1]);
  CHECK_NULL_RETURN(in_tensors_[2]);
  CHECK_NULL_RETURN(in_tensors_[3]);
  CHECK_NULL_RETURN(in_tensors_[4]);
  CHECK_NULL_RETURN(in_tensors_[5]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[1]);
  CHECK_NULL_RETURN(out_tensors_[2]);
  return ReSize();
}

int BNGradCPUKernel::DoExecute(int task_id) {
  auto *input_yt = in_tensors_.at(0);
  auto *input_x = in_tensors_.at(1);
  auto *input_scale = in_tensors_.at(2);
  auto *input_mean = in_tensors_.at(3);
  auto *input_var = in_tensors_.at(4);

  auto kernel_name = this->name();
  if (kernel_name.find("FusedBatchNormGradCPU") != std::string::npos) {
    input_mean = in_tensors_.at(4);
    input_var = in_tensors_.at(5);
  }
  auto bn_param = reinterpret_cast<BNGradParameter *>(op_parameter_);
  CHECK_NULL_RETURN(bn_param);
  int stage = stage_;
  int thread_num = thread_num_;
  float *save_mean = reinterpret_cast<float *>(input_mean->MutableData());
  CHECK_NULL_RETURN(save_mean);
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());
  CHECK_NULL_RETURN(save_var);

  auto *output_dx = out_tensors_.at(0);
  auto *output_scale = out_tensors_.at(1);
  auto *output_bias = out_tensors_.at(2);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();
  float *x = reinterpret_cast<float *>(input_x->MutableData());
  CHECK_NULL_RETURN(x);
  float *yt = reinterpret_cast<float *>(input_yt->MutableData());
  CHECK_NULL_RETURN(yt);
  float *scale = reinterpret_cast<float *>(input_scale->MutableData());
  CHECK_NULL_RETURN(scale);
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  CHECK_NULL_RETURN(dx);
  float *dbias = reinterpret_cast<float *>(output_bias->MutableData());
  CHECK_NULL_RETURN(dbias);
  float *dscale = reinterpret_cast<float *>(output_scale->MutableData());
  int total = 0;
  if (in_tensors().at(1)->shape().size() == kNumInputDim4) {
    total = spatial * batch;
  } else if (in_tensors().at(1)->shape().size() == kNumInputDim2) {
    total = batch;
    channels = input_scale->ElementsNum();
  } else {
    MS_LOG(ERROR) << "Unsupported tensor shape: " << in_tensors().at(1)->shape().size();
    return RET_ERROR;
  }
  int stride = UP_DIV(total, thread_num);
  int count = MSMIN(stride, total - stride * task_id);
  count = (count < 0) ? 0 : count;
  switch (stage) {
    case 0: {
      for (int job = task_id; job < kMaxTaskNum; job += thread_num) {
        switch (job) {
          case 0:
            var2Invar(save_var, input_var->ElementsNum(), bn_param->epsilon_);
            break;
          case 1:
            std::fill(dbias, dbias + channels, 0.f);
            break;
          case 2:
            std::fill(dscale, dscale + channels, 0.f);
            break;
          default:
            MS_LOG(ERROR) << "Exceeds the maximum thread";
            return RET_ERROR;
        }
      }
      if (thread_num == 1) {
        backwardAll(x, yt, save_mean, save_var, scale, total, channels, dbias, dscale, dx,
                    (IsTrain() && bn_param->is_training_));
      }
      break;
    }
    case 1: {
      backwardP1(x, yt, save_mean, save_var, scale, total, channels, dbias, dscale);
      break;
    }
    case 2: {
      backwardP2(x + task_id * stride * channels, yt + task_id * stride * channels, save_mean, save_var, dscale, dbias,
                 scale, count, total, channels, dx + task_id * stride * channels,
                 (IsTrain() && bn_param->is_training_));
      break;
    }
    default:
      MS_LOG(ERROR) << "Unsupported stage";
      return RET_ERROR;
  }

  return RET_OK;
}

int BNGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernel *>(cdata);
  auto error_code = bn_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernel::Run() {
  stage_ = 0;
  thread_num_ = op_parameter_->thread_num_;
  int error_code;
  if (thread_num_ == 1) {
    error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    const std::vector<int> threads = {thread_num_, 1, thread_num_};
    for (size_t stage = 0; stage < threads.size(); stage++) {
      stage_ = static_cast<int>(stage);
      error_code = ParallelLaunch(this->ms_context_, BNGradRun, this, threads.at(stage));
      if (error_code != RET_OK) {
        MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
        return RET_ERROR;
      }
    }
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBNGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNormGrad);
  auto *kernel = new (std::nothrow) BNGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BNGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNormGrad, CpuBNGradFp32KernelCreator)
}  // namespace mindspore::kernel
