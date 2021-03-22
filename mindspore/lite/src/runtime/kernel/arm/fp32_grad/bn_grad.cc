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

#include "src/runtime/kernel/arm/fp32_grad/bn_grad.h"
#include <math.h>
#include <algorithm>
#include <vector>
#include <string>
#include <thread>
#include <fstream>

#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32_grad/batch_norm.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNormGrad;

namespace mindspore::kernel {
int BNGradCPUKernel::ReSize() {
  auto *input_x = in_tensors_.at(1);
  int channels = input_x->shape().at(kNHWC_C);
  ws_size_ = 2 * channels;
  set_workspace_size(ws_size_ * sizeof(float));
  return RET_OK;
}

int BNGradCPUKernel::Init() { return ReSize(); }

int BNGradCPUKernel::Execute(int task_id) {
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
  int stage = stage_;
  int thread_num = thread_num_;
  float *save_mean = reinterpret_cast<float *>(input_mean->MutableData());
  float *save_var = reinterpret_cast<float *>(input_var->MutableData());

  auto *output_dx = out_tensors_.at(0);
  auto *output_scale = out_tensors_.at(1);
  auto *output_bias = out_tensors_.at(2);
  int32_t batch = input_x->Batch();
  int32_t channels = input_x->Channel();
  int32_t spatial = input_x->Height() * input_x->Width();

  float *workspace_temp = static_cast<float *>(workspace());
  float *dxhat_sum = workspace_temp;
  float *dxhathat_sum = dxhat_sum + channels;
  float *x = reinterpret_cast<float *>(input_x->MutableData());
  float *yt = reinterpret_cast<float *>(input_yt->MutableData());
  float *scale = reinterpret_cast<float *>(input_scale->MutableData());
  float *dx = reinterpret_cast<float *>(output_dx->MutableData());
  float *dbias = reinterpret_cast<float *>(output_bias->MutableData());
  float *dscale = reinterpret_cast<float *>(output_scale->MutableData());
  int total = spatial * batch;
  int stride = UP_DIV(total, thread_num);
  int count = MSMIN(stride, total - stride * task_id);
  count = (count < 0) ? 0 : count;
  switch (stage) {
    case 0: {
      for (int job = task_id; job < 4; job += thread_num) {
        switch (job) {
          case 0:
            var2Invar(save_var, input_var->ElementsNum(), bn_param->epsilon_);
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
        backwardAll(x, yt, save_mean, save_var, scale, total, channels, dxhat_sum, dxhathat_sum, dbias, dscale, dx);
      }
      break;
    }
    case 1: {
      backwardP1(x, yt, save_mean, save_var, scale, total, channels, dxhat_sum, dxhathat_sum, dbias, dscale);
      break;
    }
    case 2: {
      backwardP2(x + task_id * stride * channels, yt + task_id * stride * channels, save_mean, save_var, scale, count,
                 total, channels, dxhat_sum, dxhathat_sum, dx + task_id * stride * channels);
      break;
    }
  }

  return RET_OK;
}

int BNGradRun(void *cdata, int task_id) {
  MS_ASSERT(cdata != nullptr);
  auto bn_kernel = reinterpret_cast<BNGradCPUKernel *>(cdata);
  auto error_code = bn_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "BNGradRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int BNGradCPUKernel::Run() {
  stage_ = 0;
  thread_num_ = context_->thread_num_;
  if (thread_num_ == 1) {
    int error_code = ParallelLaunch(this->context_->thread_pool_, BNGradRun, this, thread_num_);
    if (error_code != RET_OK) {
      MS_LOG(ERROR) << "BN function error error_code[" << error_code << "]";
      return RET_ERROR;
    }
  } else {
    const std::vector<int> threads = {thread_num_, 1, thread_num_};
    for (size_t stage = 0; stage < threads.size(); stage++) {
      stage_ = static_cast<int>(stage);
      int error_code = ParallelLaunch(this->context_->thread_pool_, BNGradRun, this, threads.at(stage));
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
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNormGrad);
  auto *kernel = new (std::nothrow) BNGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BNGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNormGrad, CpuBNGradFp32KernelCreator)
}  // namespace mindspore::kernel
